"""
Graphite Topology Optimization - MMA Solvers

This module provides industrial-grade wrapping of the Method of Moving Asymptotes 
(MMA) via NLopt for scikit-topt optimization tasks. It includes implementations 
for standard compliance minimization as well as advanced multi-constraint stress 
P-norm minimization with Heaviside projection.
"""
import numpy as np
import nlopt
import logging
import skfem
import meshio
import os
from scipy.sparse.linalg import splu
from skfem.helpers import dot, grad
from sktopt import core

logger = logging.getLogger("MMA_Solver")

class MMA_Optimizer:
    """
    Industrial-grade MMA wrapper for scikit-topt tasks using NLopt.
    
    Uses an internal OC_Optimizer to handle FEA and Filtering consistently.

    Parameters
    ----------
    task : sktopt.Task
        The optimization task defining the mesh, boundary conditions, and loads.
    config : Any
        Configuration object containing solver parameters (e.g., target_volume_fraction, 
        max_iterations, rho_min).
    """
    def __init__(self, task, config):
        self.task = task
        self.config = config
        self.n_elements = len(task.all_elements)
        self.passive_idx = task.fixed_elements
        self.design_idx  = task.design_elements
        
        # 1. Initialize an internal OC_Optimizer to get the FEM and Filter objects
        from sktopt.tools import SchedulerConfig
        import sktopt.filters
        oc_cfg = core.OC_Config(
            vol_frac=SchedulerConfig('Step', config.target_volume_fraction, config.target_volume_fraction, 1, config.max_iterations),
            max_iters=config.max_iterations,
            filter_type='helmholtz',
            filter_radius=SchedulerConfig('Step', 4.0, 4.0, 1, config.max_iterations),
            rho_min=getattr(config, 'rho_min', 1e-6)
        )
        self.oc_opt = core.OC_Optimizer(oc_cfg, task.task)
        
        # Assign filter BEFORE state initialization
        self.oc_opt.filter = sktopt.filters.HelmholtzFilterNodal(
            mesh=task.basis.mesh,
            elements_volume=task.elements_volume,
            radius=4.0
        )
        
        self.oc_opt.initialize_density()
        if hasattr(self.oc_opt, '_ensure_state_initialized'):
            self.oc_opt._ensure_state_initialized()
        
        # Reference shortcuts
        self.fem = self.oc_opt.fem
        self.filter = self.oc_opt.filter
        
        # 2. Setup NLopt
        self.opt = nlopt.opt(nlopt.LD_MMA, self.n_elements)
        rho_min_floor = getattr(config, 'rho_min', 1e-6)
        self.opt.set_lower_bounds(np.full(self.n_elements, rho_min_floor))
        self.opt.set_upper_bounds(np.ones(self.n_elements))
        
        # Hard-lock passive elements
        lb = self.opt.get_lower_bounds()
        lb[self.passive_idx] = 1.0
        self.opt.set_lower_bounds(lb)
        
        ub = self.opt.get_upper_bounds()
        ub[self.passive_idx] = 1.0 # Also lock upper bound to 1.0
        self.opt.set_upper_bounds(ub)
        
        self.opt.set_min_objective(self._objective_callback)
        self.opt.add_inequality_constraint(self._volume_constraint, 1e-7)
        self.opt.set_maxeval(config.max_iterations)
        self.opt.set_xtol_rel(1e-8)
        
        self.iter_count = 0
        self.last_compliance = 0.0

    def _objective_callback(self, x, grad):
        self.iter_count += 1
        p = 3.0 # SIMP Penalty
        
        # Helmholtz Filter
        rho_filtered = self.filter.forward(x)
        
        # Solve FEA and get Compliance
        # u_dofs will be populated in-place
        u_dofs = np.zeros((self.task.basis.N, self.task.n_tasks))
        compliance_arr = self.fem.objectives_multi_load(
            rho_filtered, p, u_dofs, force_scale=1.0
        )
        compliance = np.sum(compliance_arr)
        
        # Calculate Energy for sensitivities
        energy = self.fem.energy_multi_load(rho_filtered, p, u_dofs)
        
        # Sensitivities (dC/d_rho_filtered)
        # For SIMP: dC/drho = -p * rho^(p-1) * energy_base = -p * (energy / rho)
        rho_safe = np.maximum(rho_filtered, 1e-6)
        dC_drho_filtered = -p * energy / rho_safe
        
        # Average sensitivities over all tasks (if multi-load)
        if dC_drho_filtered.ndim > 1:
            dC_drho_filtered = np.mean(dC_drho_filtered, axis=1)
        
        if grad.size > 0:
            # Chain Rule: dC/dx = Filter^T * dC/d_rho_filtered
            grad[:] = self._apply_filter_adjoint(dC_drho_filtered)
            
        self.last_compliance = compliance
        logger.info(f"MMA Iter {self.iter_count}: Compliance = {compliance:.4e}")
            
        return compliance

    def _volume_constraint(self, x, grad):
        volumes = self.task.elements_volume
        total_vol = np.sum(volumes)
        limit_vol = self.config.target_volume_fraction * total_vol
        current_vol = np.dot(x, volumes)
        
        if grad.size > 0:
            grad[:] = volumes / limit_vol
            
        return (current_vol / limit_vol) - 1.0

    def optimize(self):
        """
        Execute the MMA optimization loop.

        Returns
        -------
        ndarray
            The final optimized elemental density field.
        """
        logger.info(f"Starting MMA Optimization...")
        init_rho = np.ones(self.n_elements) * self.config.target_volume_fraction
        init_rho[self.passive_idx] = 1.0
        
        try:
            final_rho = self.opt.optimize(init_rho)
            return final_rho
        except Exception as e:
            logger.error(f"MMA Failed: {e}")
            return init_rho

    def _filter_element(self, mesh):
        if isinstance(mesh, skfem.MeshTri):
            return skfem.ElementTriP1()
        if isinstance(mesh, skfem.MeshQuad):
            return skfem.ElementQuad1()
        if isinstance(mesh, skfem.MeshTet):
            return skfem.ElementTetP1()
        if isinstance(mesh, skfem.MeshHex):
            return skfem.ElementHex1()
        raise ValueError(f'Unsupported mesh type for Helmholtz adjoint: {type(mesh)}')

    def _get_filter_adjoint_cache(self):
        cache = getattr(self, '_filter_adjoint_cache', None)
        if cache is not None:
            return cache

        mesh = self.filter.mesh
        basis = skfem.Basis(mesh, self._filter_element(mesh))

        @skfem.BilinearForm
        def mass(u, v, w):
            return u * v

        @skfem.BilinearForm
        def helmholtz(u, v, w):
            return u * v + (self.filter.radius ** 2) * dot(grad(u), grad(v))

        mass_matrix = skfem.asm(mass, basis).tocsc()
        operator_matrix = skfem.asm(helmholtz, basis).tocsc()
        operator_solver = splu(operator_matrix)

        node_weights = np.zeros(mesh.p.shape[1])
        for element_index, nodes in enumerate(mesh.t.T):
            node_weights[nodes] += self.filter.elements_volume[element_index]
        node_weights[node_weights == 0.0] = 1.0

        fixed_nodes = np.array([], dtype=int)
        design_mask = getattr(self.filter, 'design_mask', None)
        if design_mask is not None:
            fixed_nodes = np.unique(mesh.t[:, ~design_mask].ravel())

        cache = {
            'basis': basis,
            'mass': mass_matrix,
            'A': operator_matrix,
            'solver': operator_solver,
            'node_weights': node_weights,
            'fixed_nodes': fixed_nodes,
        }
        self._filter_adjoint_cache = cache
        return cache

    def _apply_filter_adjoint(self, arr):
        """Apply the exact transpose of the nodal Helmholtz filter.

        Forward operator: x_e -> node averaging -> A^{-1} M -> element averaging.
        Adjoint operator: element transpose -> M A^{-1} -> node transpose.
        """
        cache = self._get_filter_adjoint_cache()
        mesh = self.filter.mesh
        t = mesh.t.T
        node_grad = np.zeros(mesh.p.shape[1])

        # transpose of node_to_element_density: distribute each element sensitivity equally
        for element_index, nodes in enumerate(t):
            node_grad[nodes] += arr[element_index] / len(nodes)

        # adjoint of A^{-1} M is M A^{-1}
        if cache['fixed_nodes'].size > 0:
            rhs = node_grad.copy()
            x0 = np.zeros(cache['A'].shape[0])
            z = skfem.solve(
                *skfem.condense(cache['A'], rhs, D=cache['fixed_nodes'], x=x0)
            )
        else:
            z = cache['solver'].solve(node_grad)
        node_grad = cache['mass'] @ z

        # transpose of element_to_node_density_averaging
        elem_grad = np.zeros(mesh.t.shape[1])
        for element_index, nodes in enumerate(t):
            weights = self.filter.elements_volume[element_index] / cache['node_weights'][nodes]
            elem_grad[element_index] = np.sum(node_grad[nodes] * weights)

        passive_idx = getattr(self, 'passive_idx', None)
        if passive_idx is not None and len(passive_idx) > 0:
            elem_grad[passive_idx] = 0.0
        return elem_grad

class Stress_P_Norm_Optimizer(MMA_Optimizer):
    """
    Multi-Constraint Optimizer — Architecture: Minimize Stress, Constrain Compliance.
    
    Objective:    Minimize P-Norm Aggregated Von Mises Stress.
    Constraint 1: Volume <= Target Volume Fraction.
    Constraint 2: Compliance <= compliance_limit (1.10 * C_base).

    PASSIVE LOCK GUARANTEE:
    After every Heaviside projection, locked elements are hard-clamped to rho=1.0
    and their sensitivities are forced to 0.0 before any gradient is returned to NLopt.

    Parameters
    ----------
    task : sktopt.Task
        The optimization task.
    config : Any
        The solver configuration.
    target_stress : float, optional
        Target stress normalization factor, by default 1.0.
    p_norm : float, optional
        The P-norm exponent for stress aggregation, by default 6.0.
    q_relaxation : float, optional
        The stress relaxation exponent to prevent singularity in void regions, by default 1.0.
    beta_init : float, optional
        Initial beta value for the Heaviside projection, by default 1.0.
    beta_max : float, optional
        Maximum beta value for the Heaviside projection, by default 32.0.
    beta_ramp_every : int, optional
        Number of iterations between beta doubling, by default 25.
    compliance_limit : float, optional
        Upper bound limit for the compliance constraint, by default None.
    """
    def __init__(self, task, config, target_stress=1.0, p_norm=6.0, q_relaxation=1.0,
                 beta_init=1.0, beta_max=32.0, beta_ramp_every=25, compliance_limit=None):
        super().__init__(task, config)
        self.p_norm = p_norm
        self.q_relaxation = q_relaxation
        self.target_stress = target_stress
        self.beta = beta_init
        self.beta_max = beta_max
        self.beta_ramp_every = beta_ramp_every
        self.eta = 0.5
        # Compliance ceiling: default to 10% above a very high value if not provided
        self.compliance_limit = compliance_limit if compliance_limit is not None else 1e12
        
        # Disable ALL early-exit criteria — FORCE full maxeval=300 to allow beta ramping schedule
        self.opt.set_xtol_rel(0)
        self.opt.set_xtol_abs(0)
        self.opt.set_ftol_rel(0)  # ← DISABLED: was 1e-4 (caused premature exit at ~30 iters)
        self.opt.set_ftol_abs(0)
        
        # Objective = Minimize P-Norm Stress
        # Constraints: Volume <= VF, Compliance <= compliance_limit
        # Objective is the stress P-norm; constraints: volume and compliance
        self.opt.set_min_objective(self._stress_objective)
        self.opt.remove_inequality_constraints()
        self.opt.add_inequality_constraint(self._volume_constraint_projected, 1e-7)
        self.opt.add_inequality_constraint(self._compliance_constraint, 1e-7)

    @staticmethod
    def _heaviside_project(rho_filtered, beta, eta):
        """
        Smooth Heaviside projection (tanh formulation).
        Maps filtered densities toward 0/1 with sharpness controlled by beta.
        """
        numerator = np.tanh(beta * eta) + np.tanh(beta * (rho_filtered - eta))
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return numerator / denominator

    @staticmethod
    def _heaviside_derivative(rho_filtered, beta, eta):
        """
        Derivative of the Heaviside projection w.r.t. rho_filtered.
        """
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        sech2 = 1.0 / np.cosh(beta * (rho_filtered - eta))**2
        return beta * sech2 / denominator

    def _update_beta(self):
        if self.iter_count > 1 and (self.iter_count - 1) % self.beta_ramp_every == 0:
            old_beta = self.beta
            self.beta = min(self.beta * 2.0, self.beta_max)
            if self.beta != old_beta:
                logger.info(f"  Beta ramp: {old_beta:.1f} -> {self.beta:.1f}")

    def _get_physical_density(self, x):
        """
        FOOLPROOF DENSITY PIPELINE — called before every physics solve:
          1. Helmholtz filter the raw design variables.
          2. Apply Heaviside projection.
          3. HARD-CLAMP passive (locked) elements to rho = 1.0.
          4. ZERO the Heaviside derivative for locked elements so gradients
             can never propagate back into those zones.
        Returns: (rho_physical, rho_filtered, dH)
        """
        rho_filtered = self.filter.forward(x)
        rho_projected = self._heaviside_project(rho_filtered, self.beta, self.eta)
        dH = self._heaviside_derivative(rho_filtered, self.beta, self.eta)
        # ABSOLUTE DENSITY OVERRIDE
        rho_projected[self.passive_idx] = 1.0
        dH[self.passive_idx] = 0.0
        return rho_projected, rho_filtered, dH

    def _stress_objective(self, x, grad):
        """OBJECTIVE: Minimize P-Norm Aggregated Von Mises Stress.
        
        CRITICAL: All FEM solves MUST use rho_projected (post-Heaviside), never rho_filtered.
        Physics pipeline: x -> Helmholtz filter -> Heaviside projection -> rho_projected -> SIMP FEM -> compliance.
        """
        self.iter_count += 1
        self._update_beta()
        p_simp = 3.0
        q_relax = self.q_relaxation

        rho_projected, rho_filtered, dH = self._get_physical_density(x)

        # --- TARGETED TELEMETRY PROBE ---
        # Tracking specific elemental densities through the Heaviside schedule
        tracked_ids = [5152, 6186, 7197] 
        try:
            probe_densities = rho_projected[tracked_ids]
            print(f"Iter Probe | Load [5152]: {probe_densities[0]:.4f} | Inner [6186]: {probe_densities[1]:.4f} | Top [7197]: {probe_densities[2]:.4f}")
        except IndexError:
            print("TELEMETRY ERROR: Element ID out of bounds. Check mesh indexing.")
        # --------------------------------

        # CRITICAL: ALWAYS use rho_projected in FEM, never rho_filtered.
        # rho_projected is the physical density after Heaviside projection.
        # rho_filtered is intermediate (bypassing this would skip Heaviside penalty).
        assert rho_projected is not None, "rho_projected must not be None"
        assert rho_projected.shape == (self.n_elements,), f"Shape mismatch: {rho_projected.shape}"
        
        # Verify passive elements are locked at 1.0
        assert np.all(rho_projected[self.passive_idx] == 1.0), "Passive elements not locked to 1.0"

        u_dofs = np.zeros((self.task.basis.N, self.task.n_tasks))
        # FEM MUST receive rho_projected, which includes Heaviside penalization
        self.fem.objectives_multi_load(rho_projected, p_simp, u_dofs, force_scale=1.0)

        energy = self.fem.energy_multi_load(rho_projected, p_simp, u_dofs)
        if energy.ndim > 1:
            energy = np.mean(energy, axis=1)

        # Telemetry export: every 5 iterations, dump elemental VTU with rho_projected and Strain_Energy
        # CRITICAL: Export rho_projected (post-Heaviside), NOT rho_filtered, to match FEM input exactly.
        if (self.iter_count % 5) == 0:
            try:
                out_dir = os.path.join(os.getcwd(), 'experiments', 'scikit_topt_sandbox', 'telemetry')
                os.makedirs(out_dir, exist_ok=True)
                points = self.task.nodes
                # meshio expects a list of cell blocks like ("tetra", elements)
                cells = [("tetra", self.task.elements)]
                # CRITICAL: rho_projected is what FEM used; export it exactly as-is
                density_arr = np.asarray(rho_projected, dtype=np.float64)
                energy_arr = np.asarray(energy, dtype=np.float64)
                cell_data = {
                    'Density_Projected': [density_arr],  # Post-Heaviside projected density
                    'Strain_Energy': [energy_arr]
                }
                mesh = meshio.Mesh(points=points, cells=cells, cell_data=cell_data)
                fname = os.path.join(out_dir, f"telemetry_iter_{self.iter_count:03d}.vtu")
                meshio.write(fname, mesh, file_format="vtu")
                logger.debug(f"Telemetry VTU: {fname} | rho_proj min={rho_projected.min():.6e}, max={rho_projected.max():.6e}")
            except Exception as exc:
                logger.warning(f"Telemetry export failed (iter {self.iter_count}): {exc}")

        rho_safe = np.maximum(rho_projected, 1e-6)
        actual_stress = np.sqrt(np.maximum(energy / (rho_safe**p_simp), 0.0))
        relaxed_stress = (rho_safe**q_relax) * actual_stress

        v = self.task.elements_volume
        sum_v_sigma_p = np.sum(v * (relaxed_stress**self.p_norm))
        j_val = (sum_v_sigma_p)**(1.0 / self.p_norm)
        
        # ANTI-FRAGMENTATION PENALTY: Discourage isolated density islands
        # Penalizes densities that are far from their neighbors (disconnected components)
        # This encourages formation of connected structures instead of speckled patterns
        perimeter_weight = 0.01 * (1.0 + self.iter_count / 100.0)  # Increase penalty over iterations
        perimeter_penalty = perimeter_weight * np.sum(rho_projected * (1.0 - rho_projected))
        
        j_val_total = j_val + perimeter_penalty

        if grad.size > 0:
            dJ_dsigma = (j_val**(1.0 - self.p_norm)) * v * (relaxed_stress**(self.p_norm - 1.0))
            term1 = dJ_dsigma * q_relax * (rho_safe**(q_relax - 1.0)) * actual_stress
            stress_weight = dJ_dsigma * (rho_safe**q_relax) / np.maximum(actual_stress, 1e-6)
            term2 = (p_simp / rho_safe) * energy * stress_weight
            dJ_drho_proj = term1 - term2
            
            # ADD PERIMETER GRADIENT: penalizes gray zone (rho between 0 and 1)
            dPerim_drho = perimeter_weight * (1.0 - 2.0 * rho_projected)
            dJ_drho_proj = dJ_drho_proj + dPerim_drho
            
            # ZERO GRADIENTS FOR LOCKED ELEMENTS before and after chain rule
            dJ_drho_proj[self.passive_idx] = 0.0
            grad[:] = self._apply_filter_adjoint(dJ_drho_proj * dH)
            grad[self.passive_idx] = 0.0

        if self.iter_count % 10 == 0:
            logger.info(f"Stress Obj Iter {self.iter_count}: P-Norm={j_val:.4e} + Perim={perimeter_penalty:.4e} = {j_val_total:.4e} (beta={self.beta:.1f})")
        return j_val_total

    def _volume_constraint_projected(self, x, grad):
        """CONSTRAINT: Volume fraction <= target.
        
        Uses rho_projected (post-Heaviside) to measure actual manufactured volume.
        This ensures VTU-exported densities match what was used for volume computation.
        """
        rho_projected, rho_filtered, dH = self._get_physical_density(x)

        volumes = self.task.elements_volume
        total_vol = np.sum(volumes)
        limit_vol = self.config.target_volume_fraction * total_vol
        current_vol = np.dot(rho_projected, volumes)  # Must use rho_projected, not rho_filtered

        if grad.size > 0:
            dVol_drho_proj = volumes / limit_vol
            dVol_drho_proj[self.passive_idx] = 0.0
            grad[:] = self._apply_filter_adjoint(dVol_drho_proj * dH)
            grad[self.passive_idx] = 0.0

        return (current_vol / limit_vol) - 1.0
    
    def diagnose_filter_disconnect(self, x):
        """DIAGNOSTIC: Check if FEM input density matches exported VTU density.
        
        If compliance appears too low despite severed structure visible in VTU,
        this diagnostic will reveal if the FEM is using a different density than expected.
        
        Returns: dict with diagnostics
        """
        p_simp = 3.0
        rho_projected, rho_filtered, dH = self._get_physical_density(x)
        
        # Compute compliance using rho_projected (as _stress_objective does)
        u_dofs = np.zeros((self.task.basis.N, self.task.n_tasks))
        self.fem.objectives_multi_load(rho_projected, p_simp, u_dofs, force_scale=1.0)
        energy_proj = self.fem.energy_multi_load(rho_projected, p_simp, u_dofs)
        compliance_proj = np.sum(energy_proj)
        
        # HYPOTHETICAL: What if FEM were accidentally using rho_filtered?
        # Recompute compliance with rho_filtered to see the theoretical difference
        u_dofs_filt = np.zeros((self.task.basis.N, self.task.n_tasks))
        try:
            self.fem.objectives_multi_load(rho_filtered, p_simp, u_dofs_filt, force_scale=1.0)
            energy_filt = self.fem.energy_multi_load(rho_filtered, p_simp, u_dofs_filt)
            compliance_filt = np.sum(energy_filt)
        except:
            compliance_filt = np.nan
        
        # Return diagnostic dict
        diag = {
            'iter': self.iter_count,
            'beta': self.beta,
            'rho_filtered_range': (rho_filtered.min(), rho_filtered.max()),
            'rho_projected_range': (rho_projected.min(), rho_projected.max()),
            'compliance_with_rho_projected': compliance_proj,
            'compliance_with_rho_filtered': compliance_filt,
            'ratio_filt_to_proj': compliance_filt / compliance_proj if compliance_proj > 0 else np.nan,
        }
        
        logger.info(f"FILTER_DISCONNECT_DIAGNOSTIC:\n"
                   f"  rho_filtered: [{diag['rho_filtered_range'][0]:.3e}, {diag['rho_filtered_range'][1]:.3e}]\n"
                   f"  rho_projected: [{diag['rho_projected_range'][0]:.3e}, {diag['rho_projected_range'][1]:.3e}]\n"
                   f"  C(rho_proj): {diag['compliance_with_rho_projected']:.4e}\n"
                   f"  C(rho_filt): {diag['compliance_with_rho_filtered']:.4e}\n"
                   f"  Ratio (filt/proj): {diag['ratio_filt_to_proj']:.3f}")
        
        return diag

    def _compliance_constraint(self, x, grad):
        """CONSTRAINT: Compliance <= compliance_limit (1.10 * C_base).
        
        CRITICAL: Must evaluate FEM using rho_projected (post-Heaviside).
        If compliance appears too low despite severed structure in VTU, 
        this is the first place to check for density mismatch.
        """
        p_simp = 3.0

        rho_projected, rho_filtered, dH = self._get_physical_density(x)

        # Diagnostic: verify projection is actually being applied
        rho_filt_min, rho_filt_max = rho_filtered.min(), rho_filtered.max()
        rho_proj_min, rho_proj_max = rho_projected.min(), rho_projected.max()
        
        u_dofs = np.zeros((self.task.basis.N, self.task.n_tasks))
        # FEM MUST use rho_projected, never rho_filtered
        comp_arr = self.fem.objectives_multi_load(rho_projected, p_simp, u_dofs, force_scale=1.0)
        compliance = np.sum(comp_arr)

        if grad.size > 0:
            energy = self.fem.energy_multi_load(rho_projected, p_simp, u_dofs)
            if energy.ndim > 1:
                energy = np.mean(energy, axis=1)
            rho_safe = np.maximum(rho_projected, 1e-6)
            dC_drho_proj = -p_simp * energy / rho_safe
            # ZERO GRADIENTS FOR LOCKED ELEMENTS
            dC_drho_proj[self.passive_idx] = 0.0
            grad[:] = self._apply_filter_adjoint((dC_drho_proj / self.compliance_limit) * dH)
            grad[self.passive_idx] = 0.0

        if self.iter_count % 10 == 0:
            logger.info(f"  Compliance Constraint: {compliance:.4e} / {self.compliance_limit:.4e} "
                       f"(Iter {self.iter_count}, rho_filt: [{rho_filt_min:.3e}, {rho_filt_max:.3e}], "
                       f"rho_proj: [{rho_proj_min:.3e}, {rho_proj_max:.3e}])")

        return (compliance / self.compliance_limit) - 1.0

    def optimize(self, init_rho=None):
        """
        Execute the multi-constraint MMA optimization loop.

        Parameters
        ----------
        init_rho : ndarray, optional
            Initial density guess. If None, initialized to target volume fraction.

        Returns
        -------
        ndarray
            The final physical (Heaviside projected) elemental density field.
        """
        logger.info(f"Starting Multi-Constraint MMA (Stress Obj, Compliance Limit: {self.compliance_limit:.2e})...")
        if init_rho is None:
            init_rho = np.ones(self.n_elements) * self.config.target_volume_fraction
            init_rho[self.passive_idx] = 1.0

        try:
            final_x = self.opt.optimize(init_rho)
        except Exception as e:
            logger.error(f"MMA Failed: {e}")
            final_x = init_rho

        rho_filtered = self.filter.forward(final_x)
        rho_final = self._heaviside_project(rho_filtered, self.beta, self.eta)
        # FINAL PASSIVE LOCK: ensure the returned density field always has locked zones = 1.0
        rho_final[self.passive_idx] = 1.0
        return rho_final

class Compliance_Heaviside_Optimizer(MMA_Optimizer):
    """
    Troubleshooting Optimizer for Compliance Minimization.
    
    Objective: Minimize Compliance (Maximum Stiffness)
    Constraint: Volume <= Target Volume
    
    Uses Heaviside Projection (beta ramp) to ensure binary results.

    Parameters
    ----------
    task : sktopt.Task
        The optimization task.
    config : Any
        The solver configuration.
    beta_init : float, optional
        Initial beta value for the Heaviside projection, by default 2.0.
    beta_max : float, optional
        Maximum beta value for the Heaviside projection, by default 32.0.
    beta_ramp_every : int, optional
        Number of iterations between beta doubling, by default 25.
    """
    def __init__(self, task, config, beta_init=2.0, beta_max=32.0, beta_ramp_every=25):
        super().__init__(task, config)
        self.beta = beta_init
        self.beta_max = beta_max
        self.beta_ramp_every = beta_ramp_every
        self.eta = 0.5
        
        # Disable early stopping
        self.opt.set_xtol_rel(0)
        self.opt.set_xtol_abs(0)
        self.opt.set_ftol_rel(0)
        self.opt.set_ftol_abs(0)
        
        self.opt.set_min_objective(self._objective_callback_projected)
        self.opt.remove_inequality_constraints()
        self.opt.add_inequality_constraint(self._volume_constraint_projected, 1e-7)

    @staticmethod
    def _heaviside_project(rho_filtered, beta, eta):
        numerator = np.tanh(beta * eta) + np.tanh(beta * (rho_filtered - eta))
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return numerator / denominator

    @staticmethod
    def _heaviside_derivative(rho_filtered, beta, eta):
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        sech2 = 1.0 / np.cosh(beta * (rho_filtered - eta))**2
        return beta * sech2 / denominator

    def _get_physical_density(self, x):
        """FOOLPROOF DENSITY PIPELINE: filter -> Heaviside -> hard-clamp passives."""
        rho_filtered = self.filter.forward(x)
        rho_projected = self._heaviside_project(rho_filtered, self.beta, self.eta)
        dH = self._heaviside_derivative(rho_filtered, self.beta, self.eta)
        # ABSOLUTE DENSITY OVERRIDE
        rho_projected[self.passive_idx] = 1.0
        dH[self.passive_idx] = 0.0
        return rho_projected, rho_filtered, dH

    def _objective_callback_projected(self, x, grad):
        self.iter_count += 1
        if self.iter_count > 1 and (self.iter_count - 1) % self.beta_ramp_every == 0:
            old_beta = self.beta
            self.beta = min(self.beta * 2.0, self.beta_max)
            if self.beta != old_beta:
                logger.info(f"  Beta ramp: {old_beta:.1f} -> {self.beta:.1f}")

        p_simp = 3.0
        rho_projected, rho_filtered, dH = self._get_physical_density(x)

        u_dofs = np.zeros((self.task.basis.N, self.task.n_tasks))
        comp_arr = self.fem.objectives_multi_load(rho_projected, p_simp, u_dofs, force_scale=1.0)
        compliance = np.sum(comp_arr)

        if grad.size > 0:
            energy = self.fem.energy_multi_load(rho_projected, p_simp, u_dofs)
            if energy.ndim > 1:
                energy = np.mean(energy, axis=1)
            rho_safe = np.maximum(rho_projected, 1e-6)
            dC_drho_proj = -p_simp * energy / rho_safe
            # ZERO GRADIENTS FOR LOCKED ELEMENTS
            dC_drho_proj[self.passive_idx] = 0.0
            grad[:] = self._apply_filter_adjoint(dC_drho_proj * dH)
            grad[self.passive_idx] = 0.0

        if self.iter_count % 10 == 0:
            logger.info(f"Compliance Iter {self.iter_count}: {compliance:.4e} (beta={self.beta})")
        return compliance

    def _volume_constraint_projected(self, x, grad):
        rho_projected, rho_filtered, dH = self._get_physical_density(x)

        volumes = self.task.elements_volume
        limit_vol = self.config.target_volume_fraction * np.sum(volumes)
        current_vol = np.dot(rho_projected, volumes)

        if grad.size > 0:
            dVol_drho_proj = volumes / limit_vol
            dVol_drho_proj[self.passive_idx] = 0.0
            grad[:] = self._apply_filter_adjoint(dVol_drho_proj * dH)
            grad[self.passive_idx] = 0.0

        return (current_vol / limit_vol) - 1.0

    def optimize(self):
        """
        Execute the compliance-Heaviside MMA optimization loop.

        Returns
        -------
        ndarray
            The final physical (Heaviside projected) elemental density field.
        """
        logger.info(f"Starting Compliance-Heaviside Optimization (beta_max={self.beta_max})...")
        init_rho = np.ones(self.n_elements) * self.config.target_volume_fraction
        init_rho[self.passive_idx] = 1.0
        
        try:
            final_x = self.opt.optimize(init_rho)
        except Exception as e:
            logger.error(f"Optimization Failed: {e}")
            final_x = init_rho
        
        rho_filtered = self.filter.forward(final_x)
        rho_final = self._heaviside_project(rho_filtered, self.beta, self.eta)
        return rho_final
