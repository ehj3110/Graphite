"""
graphite/lbm/lettuce_solver.py
==============================
PyTorch/Lettuce LBM solver for permeability and Wall Shear Stress (WSS) extraction.
"""

from __future__ import annotations

import numpy as np
import torch
import lettuce as lt
from lettuce.ext import BGKCollision, BounceBackBoundary, QuadraticEquilibrium, D3Q19, Guo

class TPMSFlow(lt.Flow):
    """
    Subclass of lt.Flow defining periodic flow through a TPMS scaffold.
    Uses BounceBackBoundary on the solid mask and relies on torch.roll
    periodicity for all outer boundaries.
    """
    def __init__(
        self,
        context,
        resolution,
        units,
        solid_mask_np,
        boundary_style="periodic",
        initial_p_pu=None,
        initial_u_pu=None,
    ):
        self._initial_p_pu = initial_p_pu
        self._initial_u_pu = initial_u_pu
        super().__init__(
            context=context,
            resolution=list(resolution),
            units=units,
            stencil=D3Q19(),
            equilibrium=QuadraticEquilibrium(),
        )
        self.boundary_style = boundary_style
        # Store solid mask on the correct device as a bool tensor
        solid_t = torch.tensor(solid_mask_np, dtype=torch.bool, device=context.device)
        self._bb = lt.ext.BounceBackBoundary(solid_t)
        
        if boundary_style in ("flow_chamber", "flow_chamber_reverse"):
            # flow_chamber: +Z flow, flow_chamber_reverse: -Z flow
            z0_rho = 1.01 if boundary_style == "flow_chamber" else 1.00
            zmax_rho = 1.00 if boundary_style == "flow_chamber" else 1.01
            # Z=0 plane (outward normal -Z)
            self._inlet = lt.ext.EquilibriumOutletP(direction=[0, 0, -1], flow=self, rho_outlet=z0_rho)
            # Z=Max plane (outward normal +Z)
            self._outlet = lt.ext.EquilibriumOutletP(direction=[0, 0, 1], flow=self, rho_outlet=zmax_rho)

    def initial_pu(self):
        """Initial pressure and velocity in physical units."""
        if self._initial_u_pu is not None:
            p0 = self._initial_p_pu
            u0 = self._initial_u_pu
            if p0 is None:
                p0 = np.zeros(self.resolution, dtype=np.float32)
            return p0, u0
        p0 = np.zeros(self.resolution, dtype=np.float32)
        u0 = np.zeros([3] + list(self.resolution), dtype=np.float32)
        return p0, u0

    @property
    def pre_boundaries(self):
        return [self._bb]

    @property
    def post_boundaries(self):
        if self.boundary_style in ("flow_chamber", "flow_chamber_reverse"):
            return [self._inlet, self._outlet]
        return []


class LettuceSolver:
    """
    High-level Lettuce-CFD LBM solver interface.
    Handles unit conversions, body-force driving, convergence tracking,
    Darcy permeability extraction, and WSS computation.
    """
    def __init__(
        self,
        voxel_grid,
        Re=1.0,
        Ma=0.02,
        acceleration_z=1e-5,
        device="cuda",
        boundary_style="periodic",
        initial_p_pu=None,
        initial_u_pu=None,
        warm_start_f_path=None,
    ):
        self.voxel_grid = voxel_grid
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.context = lt.Context(device=self.device, use_native=False)
        
        # Characteristic length is the number of grid points along the Z axis (flow direction)
        nz = voxel_grid.nz
        # Characteristic length in physical units is the physical length along the Z axis (converted from mm to meters for SI consistency)
        lz_pu = voxel_grid.domain_size_mm[2] / 1000.0  # mm -> m
        
        self.units = lt.UnitConversion(
            reynolds_number=Re,
            mach_number=Ma,
            characteristic_length_lu=nz,
            characteristic_length_pu=lz_pu,
            characteristic_density_pu=1000.0,  # kg/m^3 (density of water)
        )
        
        self.tau = self.units.relaxation_parameter_lu
        if self.tau <= 0.5:
            raise ValueError(
                f"LBM relaxation parameter tau = {self.tau:.4f} <= 0.5 (stability limit).\n"
                f"Please decrease Reynolds number (Re) or increase Mach number (Ma) to raise tau."
            )
            
        self.flow = TPMSFlow(
            context=self.context,
            resolution=(voxel_grid.nx, voxel_grid.ny, voxel_grid.nz),
            units=self.units,
            solid_mask_np=voxel_grid.solid_mask,
            boundary_style=boundary_style,
            initial_p_pu=initial_p_pu,
            initial_u_pu=initial_u_pu,
        )

        if warm_start_f_path is not None:
            f_path = str(warm_start_f_path)
            self.flow.load(f_path)
            print(f"Warm start: loaded distribution f from {f_path}", flush=True)
        elif initial_u_pu is not None:
            print("Warm start: equilibrium from velocity/pressure guess", flush=True)
        
        if boundary_style in ("flow_chamber", "flow_chamber_reverse"):
            acceleration_z = 0.0
            
        self.acceleration_z = acceleration_z
        acc_vector = [0.0, 0.0, acceleration_z]
        self.force = Guo(flow=self.flow, tau=self.tau, acceleration=acc_vector)
        self.collision = BGKCollision(tau=self.tau, force=self.force)
        self.simulation = lt.Simulation(flow=self.flow, collision=self.collision, reporter=[])
        
        # Patch Lettuce bug where no_streaming_mask is incorrectly initialized with 
        # self.collision_index (which is 1), disabling streaming across the entire domain.
        # We re-initialize it to 0 and re-apply any valid masks from the boundaries.
        if hasattr(self.simulation, "no_streaming_mask") and self.simulation.no_streaming_mask is not None:
            self.simulation.no_streaming_mask.zero_()
            for boundary in self.flow.pre_boundaries:
                nsm = boundary.make_no_streaming_mask(
                    list(self.flow.f.shape), context=self.context
                )
                if nsm is not None:
                    # Convert to bool/uint8 and bitwise OR
                    nsm_t = torch.as_tensor(nsm, dtype=torch.uint8, device=self.device)
                    self.simulation.no_streaming_mask |= nsm_t
            for boundary in self.flow.post_boundaries:
                nsm = boundary.make_no_streaming_mask(
                    list(self.flow.f.shape), context=self.context
                )
                if nsm is not None:
                    nsm_t = torch.as_tensor(nsm, dtype=torch.uint8, device=self.device)
                    self.simulation.no_streaming_mask |= nsm_t

        self.steps_run = 0
        self.history_mean_uz = []
        self.history_steps = []

    def step(self, num_steps=1):
        """Advances the LBM simulation by num_steps."""
        self.simulation(num_steps)
        self.steps_run += num_steps

    def run_until_convergence(
        self,
        max_steps=5000,
        check_interval=100,
        tolerance=1e-5,
        min_steps=0,
    ) -> bool:
        """
        Runs the simulation until the relative change in mean Z-velocity is below tolerance.
        Returns True if converged, False if max_steps reached.
        """
        prev_mean_uz = None
        
        for _ in range(0, max_steps, check_interval):
            steps_to_run = min(check_interval, max_steps - self.steps_run)
            if steps_to_run <= 0:
                break
                
            self.step(steps_to_run)
            
            # Compute mean velocity in flow direction (Z) over the entire domain
            u_z = self.flow.u()[2]
            mean_uz = torch.mean(u_z).item()
            
            self.history_mean_uz.append(mean_uz)
            self.history_steps.append(self.steps_run)
            
            if prev_mean_uz is not None and abs(mean_uz) > 1e-12:
                relative_change = abs(mean_uz - prev_mean_uz) / abs(mean_uz)
                if self.steps_run % check_interval == 0:
                    print(
                        f"  step {self.steps_run}/{max_steps}: "
                        f"mean Uz={mean_uz:.3e}, rel_change={relative_change:.2e}",
                        flush=True,
                    )
                if relative_change < tolerance and self.steps_run >= min_steps:
                    print(
                        f"Converged after {self.steps_run} steps "
                        f"(min_steps={min_steps}).",
                        flush=True,
                    )
                    return True
            else:
                relative_change = float('inf')
                
            prev_mean_uz = mean_uz
            
        return False

    def compute_permeability(self) -> float:
        """
        Computes the Darcy permeability in physical units (mm^2).
        For body-force driven flow: k = U * nu / a
        For pressure-driven flow: k = U * mu * L / dP = U * nu * rho * L / dP
        """
        u_z = self.flow.u()[2]
        U_lu = torch.mean(u_z).item()
        
        # Convert to physical units
        U_pu = self.units.convert_velocity_to_pu(U_lu)
        nu_pu = self.units.viscosity_pu
        
        if self.flow.boundary_style == "flow_chamber":
            # Pressure-driven Darcy's law: k = U * nu * rho * L / dP
            rho_pu = self.units.characteristic_density_pu
            L_pu = self.units.characteristic_length_pu
            
            # Compute actual pressure drop across the Z domain
            p_field = self.get_pressure_field()
            # Average pressure at inlet (Z=0) and outlet (Z=-1)
            p_in = np.mean(p_field[:, :, 0])
            p_out = np.mean(p_field[:, :, -1])
            dP = p_in - p_out
            
            if abs(dP) < 1e-12:
                return 0.0
                
            k_m2 = (U_pu * nu_pu * rho_pu * L_pu) / dP
        else:
            # Body-force driven Darcy's law
            a_pu = self.units.convert_acceleration_to_pu(self.acceleration_z)
            if abs(a_pu) < 1e-18:
                return 0.0
            k_m2 = U_pu * nu_pu / a_pu
            
        # Convert m^2 to mm^2
        k_mm2 = k_m2 * 1e6
        return k_mm2

    def compute_wss_field(self) -> np.ndarray:
        """
        Computes the 3D physical Wall Shear Stress (WSS) magnitude field (in Pa)
        on the fluid-solid boundary interface voxels. Returns a numpy array of shape (NX, NY, NZ).
        """
        f = self.flow.f
        feq = self.flow.equilibrium(self.flow)
        df = f - feq
        e = self.flow.torch_stencil.e
        tau = self.tau
        
        # 1. Stress tensor: sigma_ab = -(1 - 1/(2*tau)) * sum_i e_ia * e_ib * df_i
        # Shape of sigma: (3, 3, NX, NY, NZ)
        sigma = - (1.0 - 0.5 / tau) * torch.einsum('qi,qj,qxyz->ijxyz', e, e, df)
        
        # 2. Smooth solid mask and calculate normal vectors
        solid_mask = torch.tensor(self.voxel_grid.solid_mask, dtype=torch.float32, device=self.device)
        phi = solid_mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, NX, NY, NZ)
        
        # circular padding to preserve periodicity
        phi_padded = torch.nn.functional.pad(phi, (1, 1, 1, 1, 1, 1), mode='circular')
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=self.device) / 27.0
        phi_smooth = torch.nn.functional.conv3d(phi_padded, kernel, padding=0)[0, 0]
        
        # gradients via central differences
        grad_x = (torch.roll(phi_smooth, shifts=-1, dims=0) - torch.roll(phi_smooth, shifts=1, dims=0)) / 2.0
        grad_y = (torch.roll(phi_smooth, shifts=-1, dims=1) - torch.roll(phi_smooth, shifts=1, dims=1)) / 2.0
        grad_z = (torch.roll(phi_smooth, shifts=-1, dims=2) - torch.roll(phi_smooth, shifts=1, dims=2)) / 2.0
        grad = torch.stack([grad_x, grad_y, grad_z], dim=0)  # shape (3, NX, NY, NZ)
        
        norm_grad = torch.norm(grad, dim=0, keepdim=True)
        n_vec = grad / torch.where(norm_grad > 1e-8, norm_grad, torch.ones_like(norm_grad))
        
        # 3. Traction vector t = sigma . n
        t_vec = torch.einsum('ijxyz,jxyz->ixyz', sigma, n_vec)  # shape (3, NX, NY, NZ)
        
        # 4. WSS vector = t - (t . n) n
        t_dot_n = torch.einsum('ixyz,ixyz->xyz', t_vec, n_vec).unsqueeze(0)
        wss_vec = t_vec - t_dot_n * n_vec
        wss_mag = torch.norm(wss_vec, dim=0)  # shape (NX, NY, NZ)
        
        # 5. Interface filter: fluid voxels that are adjacent to solid walls
        solid_padded = torch.nn.functional.pad(solid_mask.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='circular')
        kernel_all_ones = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=self.device)
        near_solid = torch.nn.functional.conv3d(solid_padded, kernel_all_ones, padding=0)[0, 0] > 0.1
        
        fluid_mask = ~torch.tensor(self.voxel_grid.solid_mask, dtype=torch.bool, device=self.device)
        interface_mask = fluid_mask & near_solid
        
        wss_mag_filtered = torch.where(interface_mask, wss_mag, torch.zeros_like(wss_mag))
        
        # 6. Convert to physical units (Pa)
        p_pu_char = self.units.characteristic_pressure_pu
        p_lu_char = self.units.characteristic_pressure_lu
        wss_mag_pu = wss_mag_filtered * (p_pu_char / p_lu_char)
        
        return wss_mag_pu.cpu().numpy()

    def get_velocity_field(self) -> np.ndarray:
        """Returns the 3D physical velocity field of shape (3, NX, NY, NZ) in m/s."""
        u_lu = self.flow.u()
        u_pu = self.units.convert_velocity_to_pu(u_lu)
        return u_pu.cpu().numpy()

    def get_pressure_field(self) -> np.ndarray:
        """Returns the 3D physical pressure field of shape (NX, NY, NZ) in Pa."""
        rho_lu = self.flow.rho()
        p_pu = self.units.convert_density_lu_to_pressure_pu(rho_lu)
        if p_pu.shape[0] == 1:
            p_pu = p_pu.squeeze(0)
        return p_pu.cpu().numpy()
