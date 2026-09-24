"""
Standalone Topology Optimization Solver using scikit-topt.
This module is strictly isolated from Aristo and app.py to prevent breaking existing functionality.
"""
import numpy as np
import trimesh
import warnings
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_optimizer_densities(opt):
    """Return the most useful density field exposed by a scikit-topt optimizer."""
    state = getattr(opt, "_state", None)
    if state is not None:
        for field in ("rho_projected", "rho_filtered", "rho"):
            densities = getattr(state, field, None)
            if densities is not None:
                return np.asarray(densities)

    densities = getattr(opt, "_rho_e_buffer", None)
    if densities is not None:
        return np.asarray(densities)

    return None

try:
    import skfem
    import sktopt
    from sktopt import core, mesh
    SKTOPT_AVAILABLE = True
except ImportError:
    SKTOPT_AVAILABLE = False
    warnings.warn("scikit-topt is not installed. Topology optimization will be unavailable.")

from graphite.aristo.aristo_solver import _generate_fea_mesh
from graphite.aristo.boundary_detection import map_surfaces_to_fea

class ToptConfig:
    """
    Configuration parameters for a Topology Optimization run.

    Parameters
    ----------
    fea_mesh_resolution : float, optional
        Target edge length for the FEA tetrahedral mesh (mm), by default 2.0.
    target_volume_fraction : float, optional
        Target volume fraction of the design domain (0.0 to 1.0), by default 0.5.
    max_iterations : int, optional
        Maximum number of optimization iterations, by default 50.
    fixed_face_ids : list of int, optional
        IDs of faces to apply fixed boundary conditions, by default None.
    load_face_ids : list of int, optional
        IDs of faces to apply loads, by default None.
    load_type : str, optional
        Type of load ('Directional', 'Normal (Push)', 'Normal (Pull)'), by default "Directional".
    load_direction : tuple of float, optional
        Global vector direction for the load (x, y, z), by default (0.0, 0.0, -1.0).
    load_magnitude : float, optional
        Total magnitude of the load applied, by default 100.0.
    youngs_modulus : float, optional
        Young's Modulus of the material (MPa), by default 1000.0.
    poisson_ratio : float, optional
        Poisson's ratio of the material, by default 0.3.
    bolt_centroid : tuple of float, optional
        XYZ coordinate of a bolt restriction zone, by default None.
    bolt_radius : float, optional
        Radius of the bolt restriction zone (mm), by default None.
    fixed_face_indices : list of int, optional
        Raw STL triangle indices for fixed faces, by default None.
    load_face_indices : list of int, optional
        Raw STL triangle indices for loaded faces, by default None.
    filter_radius : float, optional
        Radius for the Helmholtz density filter (mm), by default 4.0.
    passive_zone_depth : int, optional
        Number of element layers to lock solid at boundaries, by default 2.
    prevent_hollowing : bool, optional
        Legacy toggle (no longer actively enforces floor), by default True.
    beta_init : float, optional
        Initial Heaviside sharpness parameter, by default 2.0.
    rho_min : float, optional
        Lower bound for elemental density (void regions), by default 1e-4.
    beta_growth : float, optional
        Growth multiplier for the Heaviside beta per cycle, by default 1.2.
    """
    def __init__(
        self,
        fea_mesh_resolution: float = 2.0,
        target_volume_fraction: float = 0.5,
        max_iterations: int = 50,
        fixed_face_ids: list[int] = None,
        load_face_ids: list[int] = None,
        load_type: str = "Directional",
        load_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
        load_magnitude: float = 100.0,
        youngs_modulus: float = 1000.0,
        poisson_ratio: float = 0.3,
        bolt_centroid: tuple[float, float, float] = None,
        bolt_radius: float = None,
        fixed_face_indices: list[int] = None,
        load_face_indices: list[int] = None,
        filter_radius: float = 4.0,  # Must be >= 2x mesh element size to avoid checkerboard
        passive_zone_depth: int = 2,    # Element-layers deep to lock solid at fixed/load faces
        prevent_hollowing: bool = True, # Kept for API compat; no longer uses density floor
        beta_init: float = 2.0,         # Initial Heaviside sharpness (2=soft, higher=sharper)
        rho_min: float = 1e-4,          # Lowered to reduce stiffness bias in void regions
        beta_growth: float = 1.2,       # Growth factor for Heaviside beta per cycle
    ):
        self.fea_mesh_resolution = fea_mesh_resolution
        self.target_volume_fraction = target_volume_fraction
        self.max_iterations = max_iterations
        self.fixed_face_ids = fixed_face_ids or []
        self.load_face_ids = load_face_ids or []
        self.fixed_face_indices = fixed_face_indices or []
        self.load_face_indices = load_face_indices or []
        self.load_type = load_type
        self.load_direction = load_direction
        self.load_magnitude = load_magnitude
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.bolt_centroid = bolt_centroid
        self.bolt_radius = bolt_radius
        self.filter_radius = filter_radius
        self.passive_zone_depth = passive_zone_depth
        self.prevent_hollowing = prevent_hollowing
        self.beta_init = beta_init

def create_sktopt_task(mesh_obj: trimesh.Trimesh, config: ToptConfig):
    """
    Generates the gmsh volumetric mesh and converts it to a scikit-fem basis.
    Returns the sktopt LinearElasticity task object.
    """
    if not SKTOPT_AVAILABLE:
        raise RuntimeError("scikit-topt is required for this operation.")
        
    logger.info("=== STARTING PHASE 1: FEA MESHING ===")
    t0 = time.time()
    # Pass structural face indices to seed the Distance field correctly (Expert Fix)
    seed_indices = config.fixed_face_indices + config.load_face_indices
    nodes, elements, surface_faces, _ = _generate_fea_mesh(
        mesh_obj, config.fea_mesh_resolution, seed_face_indices=seed_indices
    )
    t1 = time.time()
    logger.info(f"-> Gmsh finished in {t1-t0:.2f} seconds. Nodes: {nodes.shape[0]}, Elements: {elements.shape[0]}")
    
    logger.info("=== STARTING PHASE 2: SCIFEM BASIS GENERATION ===")
    t2 = time.time()
    m = skfem.MeshTet(nodes.T, elements.T)
    # FIX: Use ElementVector to avoid 3D array broadcasting crashes in scikit-topt
    basis = skfem.Basis(m, skfem.ElementVector(skfem.ElementTetP1()))
    t3 = time.time()
    logger.info(f"-> scikit-fem basis generated in {t3-t2:.2f} seconds. Total DOFs: {basis.N}")
    
    # Map fixed_face_ids and load_face_ids to nodes using Aristo's logic
    if config.fixed_face_indices or config.load_face_indices:
        logger.info(f"Using raw STL Face Indices from Vedo. Fixed: {len(config.fixed_face_indices)}, Load: {len(config.load_face_indices)}")
        import trimesh
        from scipy.spatial import cKDTree
        
        # We need the original raw STL to get the actual face coordinates
        # Fortunately, the `mesh_obj` passed in is the trimesh!
        tm = mesh_obj
        
        # Build KDTree of FEA nodes
        fea_tree = cKDTree(nodes)
        
        # Robustly find all nodes on these large faces using proximity queries
        # (instead of just looking near vertices, which fails for large triangles)
        from trimesh.proximity import ProximityQuery
        prox = ProximityQuery(tm)
        
        # Get closest face and distance for every FEA node
        # This can be slow for huge meshes, but for ~10k nodes it's fine.
        _, distances, face_indices = prox.on_surface(nodes)
        
        # Use a tolerance (e.g. 10% of mesh resolution)
        tol = config.fea_mesh_resolution * 0.1
        
        fixed_nodes = np.where((distances < tol) & np.isin(face_indices, config.fixed_face_indices))[0]
        load_nodes = np.where((distances < tol) & np.isin(face_indices, config.load_face_indices))[0]
        
        logger.info(f"Proximity Search Found: {len(fixed_nodes)} fixed nodes, {len(load_nodes)} load nodes.")
        
    elif not config.fixed_face_ids and not config.load_face_ids:
        logger.info("No surface IDs provided. Falling back to automatic Z-bounds heuristic.")
        z_min = nodes[:, 2].min()
        z_max = nodes[:, 2].max()
        tol = (z_max - z_min) * 0.05
        fixed_nodes = np.where(nodes[:, 2] <= z_min + tol)[0]
        load_nodes = np.where(nodes[:, 2] >= z_max - tol)[0]
    else:
        logger.info(f"Mapping fixed IDs: {config.fixed_face_ids} and load IDs: {config.load_face_ids}")
        fea_surface_ids = map_surfaces_to_fea(mesh_obj, nodes, surface_faces)
        
        fixed_face_mask = np.zeros(surface_faces.shape[0], dtype=bool)
        for fid in config.fixed_face_ids:
            fixed_face_mask |= (fea_surface_ids == fid)
            
        load_face_mask = np.zeros(surface_faces.shape[0], dtype=bool)
        for fid in config.load_face_ids:
            load_face_mask |= (fea_surface_ids == fid)
            
        fixed_nodes = np.unique(surface_faces[fixed_face_mask].ravel())
        load_nodes = np.unique(surface_faces[load_face_mask].ravel())
        
        if config.bolt_centroid is not None and config.bolt_radius is not None:
            logger.info(f"Applying Bolt Area Restriction at {config.bolt_centroid} with radius {config.bolt_radius}")
            centroid_arr = np.array(config.bolt_centroid)
            distances = np.linalg.norm(nodes[load_nodes] - centroid_arr, axis=1)
            load_nodes = load_nodes[distances <= config.bolt_radius]
            logger.info(f"Filtered load nodes down to {len(load_nodes)} nodes within bolt radius.")

    M = elements.shape[0]
    all_elems = np.arange(M)
    
    # FIX: Extract flattened DOF array instead of using [0,1,2] for dirichlet
    t4 = time.time()
    fixed_dofs = basis.nodal_dofs[:, fixed_nodes].flatten()
    
    # FIX: Construct full DOF-sized force vector to prevent internal sktopt build errors
    # Distribute the load across the specified direction
    f = np.zeros(basis.N)
    
    if len(load_nodes) > 0:
        f_per_node = config.load_magnitude / len(load_nodes)
        
        if config.load_type == "Directional":
            dir_vec = np.array(config.load_direction)
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > 1e-12:
                dir_hat = dir_vec / dir_norm
            else:
                dir_hat = np.array([0.0, 0.0, -1.0])
                
            f[basis.nodal_dofs[0, load_nodes]] = f_per_node * dir_hat[0]
            f[basis.nodal_dofs[1, load_nodes]] = f_per_node * dir_hat[1]
            f[basis.nodal_dofs[2, load_nodes]] = f_per_node * dir_hat[2]
            
        else:
            # Normal (Push) or Normal (Pull)
            from scipy.spatial import cKDTree
            logger.info(f"Applying Normal Forces ({config.load_type}) to load nodes...")
            
            # We need the normals and centroids of the specific raw STL faces the user selected
            selected_normals = mesh_obj.face_normals[config.load_face_indices]
            selected_centroids = mesh_obj.triangles_center[config.load_face_indices]
            
            # Map each FEA load node to the closest selected STL face to inherit its normal
            normal_tree = cKDTree(selected_centroids)
            _, closest_face_idxs = normal_tree.query(nodes[load_nodes])
            
            mapped_normals = selected_normals[closest_face_idxs]
            
            # Normalize them to ensure unit vectors
            norms = np.linalg.norm(mapped_normals, axis=1)
            mapped_normals[norms > 0] /= norms[norms > 0, np.newaxis]
            
            # Push = pointing inwards (opposite of outward surface normal)
            if config.load_type == "Normal (Push)":
                mapped_normals = -mapped_normals
                
            f[basis.nodal_dofs[0, load_nodes]] = f_per_node * mapped_normals[:, 0]
            f[basis.nodal_dofs[1, load_nodes]] = f_per_node * mapped_normals[:, 1]
            f[basis.nodal_dofs[2, load_nodes]] = f_per_node * mapped_normals[:, 2]

        # scikit-topt uses the dot product sum of neumann_values for volume scaling internally? 
        # Wait, scikit-topt requires neumann_values to just exist so it doesn't crash, but it uses neumann_linear.
        # We will pass load_magnitude directly to neumann_values as a dummy so it passes validation.

    # Derive neumann_dir_type string from the actual load direction so the
    # optimizer's internal sensitivity uses the correct physics.
    # scikit-topt expects 'u^1', 'u^2', 'u^3' format.
    if config.load_type == "Directional":
        dir_vec = np.array(config.load_direction)
        dominant_axis = int(np.argmax(np.abs(dir_vec)))
        neumann_dir_str = ['u^1', 'u^2', 'u^3'][dominant_axis]
    else:
        # For normal forces the directions are mixed per-node; use the dominant
        # component of the average normal as a best approximation for this field.
        neumann_dir_str = 'u^3'  # fallback
    logger.info(f"Using neumann_dir_type: {neumann_dir_str}")

    # ── Passive solid element designation ─────────────────────────────────────
    # Starting from fixed/load nodes, we expand the passive zone outward by
    # config.passive_zone_depth element-layers using BFS through the element
    # connectivity graph. This creates a solid 'pad' at every BC surface that
    # the optimizer cannot carve away, guaranteeing a watertight topology.
    bc_nodes = np.union1d(fixed_nodes, load_nodes)

    # Build element adjacency: for each node, which elements contain it?
    from collections import defaultdict
    node_to_elems = defaultdict(set)
    for eid, tet in enumerate(elements):
        for nid in tet:
            node_to_elems[nid].add(eid)

    # BFS expansion from bc_nodes outward for passive_zone_depth layers
    passive_set = set()
    frontier_nodes = set(bc_nodes.tolist())
    for _layer in range(config.passive_zone_depth):
        new_elems = set()
        for nid in frontier_nodes:
            new_elems.update(node_to_elems[nid])
        new_elems -= passive_set
        passive_set.update(new_elems)
        # Next frontier: all nodes of newly added elements
        new_nodes = set()
        for eid in new_elems:
            new_nodes.update(elements[eid].tolist())
        frontier_nodes = new_nodes - set(bc_nodes.tolist())

    passive_elements = np.array(sorted(passive_set), dtype=int)
    design_elements  = np.array([e for e in all_elems if e not in passive_set], dtype=int)
    
    # ── Physical Volume Calculation ──────────────────────────────────────────
    # Calculate geometric volume of each tetrahedron for proper volume constraints
    t = basis.mesh.t
    p = basis.mesh.p
    v1, v2, v3, v4 = p[:, t[0]], p[:, t[1]], p[:, t[2]], p[:, t[3]]
    mats = np.empty((t.shape[1], 3, 3))
    mats[:, 0, :] = (v1 - v4).T
    mats[:, 1, :] = (v2 - v4).T
    mats[:, 2, :] = (v3 - v4).T
    element_volumes = np.abs(np.linalg.det(mats)) / 6.0
    
    logger.info(f"Passive zone depth={config.passive_zone_depth}: "
                f"{len(passive_elements)} locked / {len(design_elements)} free / {M} total")
    logger.info(f"Physical Volume: Total={np.sum(element_volumes):.1f}, Passive={np.sum(element_volumes[passive_elements]):.1f}")

    logger.info("=== STARTING PHASE 3: LINEAR ELASTICITY TASK GENERATION ===")
    task = mesh.LinearElasticity(
        basis=basis,
        dirichlet_nodes=[fixed_nodes],
        dirichlet_dofs=fixed_dofs,
        dirichlet_elements=None,
        dirichlet_values=[0.0],
        neumann_nodes=[load_nodes],
        neumann_elements=None,
        neumann_dir_type=[neumann_dir_str],
        neumann_values=[config.load_magnitude],
        robin_facets_ids=None,
        robin_nodes=None,
        robin_elements=None,
        robin_coefficient=None,
        robin_bc_value=None,
        design_robin_boundary=None,
        design_elements=design_elements,
        free_dofs=np.array([], dtype=int),
        free_elements=design_elements,
        all_elements=all_elems,
        fixed_elements=passive_elements,
        dirichlet_neumann_elements=passive_elements,
        elements_volume=element_volumes,
        E=config.youngs_modulus,
        nu=config.poisson_ratio,
        neumann_linear=[f]
    )
    t5 = time.time()
    logger.info(f"-> Task generation finished in {t5-t4:.2f} seconds.")
    logger.info(f"-> TOTAL SETUP TIME: {t5-t0:.2f} seconds.")
    
    # Attach metadata to task for diagnostic reporting in the UI
    task._passive_count = len(passive_elements)
    task._design_count  = len(design_elements)
    task._total_count   = M
    task._passive_fraction = len(passive_elements) / M

    return task, basis, nodes

def run_topt_optimization(task, config: ToptConfig):
    """
    Configures and runs the scikit-topt optimizer.
    """
    if not SKTOPT_AVAILABLE:
        raise RuntimeError("scikit-topt is required for this operation.")
        
    # ── Volume fraction correction for passive elements ───────────────────────
    # The OC optimizer applies vol_frac to ALL elements. Since passive elements
    # are locked at rho=1, they already consume part of the volume budget.
    # We compute an adjusted fraction so that the user's target applies to
    # the entire part, not just the free design elements.
    N_total   = getattr(task, '_total_count',   len(task.all_elements))
    N_passive = getattr(task, '_passive_count',  0)
    N_design  = getattr(task, '_design_count',  len(task.design_elements))
    passive_frac = N_passive / N_total

    # Effective fraction of DESIGN elements needed to hit the global target
    if N_design > 0:
        adj_vol = (config.target_volume_fraction * N_total - N_passive) / N_design
    else:
        adj_vol = config.target_volume_fraction

    if adj_vol <= 0.05:
        logger.warning(
            f"[!] Target volume fraction ({config.target_volume_fraction*100:.0f}%) is BELOW the "
            f"passive zone size ({passive_frac*100:.0f}%). "
            f"Optimizer will keep almost no free material. "
            f"Raise target_vol above {passive_frac*100:.0f}% or reduce passive_zone_depth."
        )
       # Correct vol_frac for design elements vs total volume
    n_p = task._passive_count
    n_d = task._design_count
    n_t = task._total_count
    
    # ── Physical Volume Calculation ──────────────────────────────────────────
    # CRITICAL: We must use the actual geometric volume of each tetrahedron.
    # These are already attached to the task as elements_volume.
    volumes = task.elements_volume
    total_mesh_vol = np.sum(volumes)
    
    passive_vol = np.sum(volumes[task.fixed_elements])
    design_vol  = np.sum(volumes[task.design_elements])
    
    # Re-calculate adj_vol based on PHYSICAL volume, not count
    adj_vol = (config.target_volume_fraction * total_mesh_vol - passive_vol) / max(design_vol, 1e-9)
    adj_vol = max(0.15, min(0.95, adj_vol))
    
    logger.info(f"PHYSICAL Volume: Total={total_mesh_vol:.1f}, Passive={passive_vol:.1f} ({passive_vol/total_mesh_vol*100:.1f}%), "
                f"DesignBudget={adj_vol*100:.1f}% of design space")

    # ── Beta (Heaviside projection) continuation ──────────────────────────────
    # Expert Advice: Stay at beta=1.0 for ~20-30 iterations, then slow ramp.
    # We lock Coarse Phase (low iters) to 1.0 to ensure connectivity.
    if config.max_iterations <= 20:
        beta_init = 1.0
        beta_target = 1.0
        beta_steps = 1
        logger.info(f"Beta schedule: LOCKED at 1.0 (Linear SIMP) for Coarse Phase")
    else:
        beta_init = 1.0
        beta_target = 8.0
        beta_steps = 8
        logger.info(f"Beta schedule: 1.0 (Linear) -> {beta_target} (Heaviside) over {config.max_iterations} iters")

    cfg = core.OC_Config(
        vol_frac=sktopt.tools.SchedulerConfig(
            scheduler_type='Step', 
            init_value=adj_vol, 
            target_value=adj_vol, 
            num_steps=1, 
            iters_max=config.max_iterations
        ),
        # elements_volume=volumes,  # OC_Config doesn't take this; it uses Task/Filter's volumes
        solver_option='cg_pyamg',
        max_iters=config.max_iterations,
        filter_type='helmholtz',
        filter_radius=sktopt.tools.SchedulerConfig(
            scheduler_type='Step', 
            init_value=config.filter_radius, 
            target_value=config.filter_radius, 
            num_steps=1, 
            iters_max=config.max_iterations
        ),
        beta=sktopt.tools.SchedulerConfig(
            scheduler_type='Step',
            init_value=beta_init,
            target_value=beta_target,
            num_steps=beta_steps,
            iters_max=240 # STRETCHED: Each step is now 30 iterations (240/8)
        ),
        rho_min=1e-3, # Lowered from 0.05 per expert advice
        export_img=False,
        record_times=min(config.max_iterations, 1)
    )
    cfg.export = lambda *args: None
    
    opt = core.OC_Optimizer(cfg, task)
    
    if not hasattr(opt, 'filter'):
        opt.filter = sktopt.filters.HelmholtzFilterNodal(
            mesh=task.basis.mesh,
            elements_volume=volumes, # Pass real volumes to filter
            radius=config.filter_radius
        )

    # ── Force passive element densities to 1.0 ──────────────────────────────────
    # The optimizer initializes ALL elements at vol_frac. Passive elements are in
    # fixed_elements so the OC update loop skips them, but they stay at vol_frac
    # (≈0.5) forever. We must:
    #   (a) Set them to 1.0 NOW, before optimization starts
    #   (b) Re-enforce after every iteration, because the Helmholtz filter blurs
    #       neighboring design-element densities into passive zone
    passive_idx = task.fixed_elements
    if len(passive_idx) > 0:
        for field in ('rho', 'rho_filtered', 'rho_projected'):
            arr = getattr(opt._state, field, None)
            if arr is not None:
                arr[passive_idx] = 1.0
        logger.info(f"Initialized {len(passive_idx)} passive elements to rho=1.0")

        # Wrap optimize() to re-clamp passive elements after every iteration
        _orig_optimize = opt.optimize
        def _optimize_with_passive_lock():
            _orig_optimize()
            for field in ('rho', 'rho_filtered', 'rho_projected'):
                arr = getattr(opt._state, field, None)
                if arr is not None:
                    arr[passive_idx] = 1.0
        opt.optimize = _optimize_with_passive_lock

    return opt


def warmstart_from_coarse(coarse_nodes: np.ndarray,
                          coarse_elements: np.ndarray,
                          coarse_densities: np.ndarray,
                          fine_nodes: np.ndarray,
                          fine_elements: np.ndarray,
                          fine_opt,
                          fine_target_vol: float) -> None:
    """
    Initialises the fine-mesh optimizer density field by interpolating the
    coarse-mesh optimized densities onto the fine mesh.

    This implements coarse->fine (multigrid continuation):
      - The coarse run quickly finds the gross topology (which regions are solid/void).
      - Interpolating to the fine mesh gives the fine optimizer a warm start that
        already encodes the load path, so it converges in far fewer iterations.

    Parameters
    ----------
    coarse_nodes     : (N_c, 3) node positions of coarse mesh
    coarse_elements  : (M_c, 4) tet connectivity of coarse mesh
    coarse_densities : (M_c,)   per-element densities from coarse optimizer
    fine_nodes       : (N_f, 3) node positions of fine mesh
    fine_elements    : (M_f, 4) tet connectivity of fine mesh
    fine_opt         : OC_Optimizer instance to warm-start
    fine_target_vol  : volume fraction to enforce when rescaling interpolated densities
    """
    from scipy.spatial import cKDTree

    # Compute centroids of each mesh's elements
    coarse_centroids = coarse_nodes[coarse_elements].mean(axis=1)  # (M_c, 3)
    fine_centroids   = fine_nodes[fine_elements].mean(axis=1)       # (M_f, 3)

    # Nearest-neighbor interpolation: each fine element gets the density of the
    # closest coarse element centroid
    tree = cKDTree(coarse_centroids)
    _, idx = tree.query(fine_centroids, k=1)
    interpolated = coarse_densities[idx]  # (M_f,)

    # Rescale so that the mean density matches the target (preserves volume constraint)
    current_mean = interpolated.mean()
    if current_mean > 1e-6:
        interpolated = interpolated * (fine_target_vol / current_mean)
    interpolated = np.clip(interpolated, 0.01, 1.0)

    logger.info(
        f"Warmstart: interpolated {len(coarse_densities)} coarse -> {len(fine_elements)} fine elements. "
        f"Mean density: {interpolated.mean():.3f} (target: {fine_target_vol:.3f})"
    )

    # Inject into optimizer state (all density fields)
    for field in ('rho', 'rho_filtered', 'rho_projected'):
        arr = getattr(fine_opt._state, field, None)
        if arr is not None:
            arr[:] = interpolated
