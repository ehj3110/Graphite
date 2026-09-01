import numpy as np
import taichi as ti
from graphite.lbm.d3q19 import Q, c, w, opposite

# Convert arrays to static tuples for JIT compiler optimization
c_tuple = tuple(tuple(int(val) for val in row) for row in c)
w_tuple = tuple(float(val) for val in w)
opposite_tuple = tuple(int(val) for val in opposite)


def _init_taichi_runtime(arch_str: str):
    """Safely initialize Taichi with the specified backend."""
    arch = ti.cpu
    if arch_str.lower() == "cuda":
        arch = ti.cuda
    elif arch_str.lower() == "opengl":
        arch = ti.opengl
    elif arch_str.lower() == "vulkan":
        arch = ti.vulkan

    # Check if already initialized
    try:
        runtime = ti.lang.impl.get_runtime()
        if runtime is not None and runtime.prog is not None:
            # Already initialized
            return
    except Exception:
        pass

    ti.init(arch=arch)


@ti.data_oriented
class LBMSolver:
    """
    A memory-efficient 3D Lattice Boltzmann Method (LBM) solver using the
    single-copy A-A streaming pattern, custom Structure of Arrays (SoA) layout,
    and halfway bounce-back boundary conditions. Optimized for GPUs with
    restricted VRAM (like the NVIDIA Mobile GTX 3050).
    """

    def __init__(
        self,
        voxel_grid,
        tau: float,
        arch: str = "cuda",
        force: tuple[float, float, float] = (0.0, 0.0, 1e-5),
    ):
        """
        Parameters
        ----------
        voxel_grid : VoxelGrid
            Container with the voxelized TPMS geometry and dimensions.
        tau : float
            LBM relaxation time. Must be > 0.5 for numerical stability.
        arch : str
            Taichi execution backend ('cuda', 'cpu', etc.).
        force : tuple of float
            Lattice unit driving acceleration vector (force per unit mass).
        """
        # 1. Validation Checks
        if tau <= 0.5:
            raise ValueError(
                f"LBM relaxation time tau must be strictly greater than 0.5 "
                f"for numerical stability. Got: {tau}"
            )

        # Enforce periodic boundary validation on Z-axis (flow direction)
        solid_mask_cpu = voxel_grid.solid_mask
        if not np.array_equal(solid_mask_cpu[:, :, 0], solid_mask_cpu[:, :, -1]):
            raise ValueError(
                "Boundary Condition Safety Violation: The solid mask is not periodic "
                "along the Z-axis (flow direction). Inlet and outlet faces must match."
            )

        # 2. Store Parameters
        self.nx = voxel_grid.nx
        self.ny = voxel_grid.ny
        self.nz = voxel_grid.nz
        self.tau = float(tau)
        self.force = (float(force[0]), float(force[1]), float(force[2]))
        self.step_count = 0

        # 3. Initialize Taichi Runtime
        _init_taichi_runtime(arch)

        # 4. Declare Fields with SoA (Structure of Arrays) Layout
        # Distribution functions: self.f[i, x, y, z]
        self.f = ti.field(dtype=ti.f32)
        ti.root.dense(ti.i, Q).dense(ti.jkl, (self.nx, self.ny, self.nz)).place(self.f)

        # Solid mask: self.solid_mask[x, y, z]
        self.solid_mask = ti.field(dtype=ti.i32)
        ti.root.dense(ti.ijk, (self.nx, self.ny, self.nz)).place(self.solid_mask)

        # Macroscopic density: self.rho[x, y, z]
        self.rho = ti.field(dtype=ti.f32)
        ti.root.dense(ti.ijk, (self.nx, self.ny, self.nz)).place(self.rho)

        # Macroscopic velocity: self.u[d, x, y, z] (optimal SoA for vector components)
        self.u = ti.field(dtype=ti.f32)
        ti.root.dense(ti.i, 3).dense(ti.jkl, (self.nx, self.ny, self.nz)).place(self.u)

        # 5. Populate Solid Mask & Initialize Fields
        self.solid_mask.from_numpy(solid_mask_cpu.astype(np.int32))
        self.init_fields()

    def step(self):
        """Execute a single time step of the LBM simulation."""
        fx, fy, fz = self.force
        if self.step_count % 2 == 0:
            self._step_even(fx, fy, fz)
        else:
            self._step_odd(fx, fy, fz)
        self.step_count += 1

    @ti.kernel
    def init_fields(self):
        """Initialize simulation fields to uniform density = 1.0 and zero velocity."""
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            self.rho[x, y, z] = 1.0
            self.u[0, x, y, z] = 0.0
            self.u[1, x, y, z] = 0.0
            self.u[2, x, y, z] = 0.0

            # Initialize f to equilibrium state for rho=1.0, u=0.0 (wi * rho)
            for i in ti.static(range(Q)):
                wi = w_tuple[i]
                self.f[i, x, y, z] = wi * 1.0

    @ti.kernel
    def _step_even(self, force_x: ti.f32, force_y: ti.f32, force_z: ti.f32):
        """
        Even step: Collision + In-place Swap.
        Reads locally, collides, and writes back locally to the opposite index.
        """
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.solid_mask[x, y, z] == 0:
                # 1. Read all local f values into local registers
                f_local = ti.Vector([0.0] * Q)
                for i in ti.static(range(Q)):
                    f_local[i] = self.f[i, x, y, z]

                # 2. Compute macroscopic density
                rho_val = 0.0
                for i in ti.static(range(Q)):
                    rho_val += f_local[i]
                self.rho[x, y, z] = rho_val

                # 3. Compute macroscopic momentum (raw velocity components)
                ux_val = 0.0
                uy_val = 0.0
                uz_val = 0.0
                for i in ti.static(range(Q)):
                    cx, cy, cz = c_tuple[i]
                    ux_val += f_local[i] * cx
                    uy_val += f_local[i] * cy
                    uz_val += f_local[i] * cz

                ux = 0.0
                uy = 0.0
                uz = 0.0
                if rho_val > 1e-8:
                    ux = ux_val / rho_val
                    uy = uy_val / rho_val
                    uz = uz_val / rho_val

                # Store physical velocity (with 0.5 * force shift)
                self.u[0, x, y, z] = ux + 0.5 * force_x
                self.u[1, x, y, z] = uy + 0.5 * force_y
                self.u[2, x, y, z] = uz + 0.5 * force_z

                # 4. Compute force-shifted velocity for equilibrium calculation
                ux_eq = ux + self.tau * force_x
                uy_eq = uy + self.tau * force_y
                uz_eq = uz + self.tau * force_z
                u2 = ux_eq * ux_eq + uy_eq * uy_eq + uz_eq * uz_eq

                # 5. Perform collision & write in-place to opposite index
                for i in ti.static(range(Q)):
                    cx, cy, cz = c_tuple[i]
                    wi = w_tuple[i]
                    opp_i = opposite_tuple[i]

                    cu = cx * ux_eq + cy * uy_eq + cz * uz_eq
                    feq = wi * rho_val * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

                    self.f[opp_i, x, y, z] = f_local[i] - (f_local[i] - feq) / self.tau
            else:
                # Solid wall macro fields
                self.rho[x, y, z] = 1.0
                self.u[0, x, y, z] = 0.0
                self.u[1, x, y, z] = 0.0
                self.u[2, x, y, z] = 0.0

    @ti.kernel
    def _step_odd(self, force_x: ti.f32, force_y: ti.f32, force_z: ti.f32):
        """
        Odd step: Pull streaming + Collision + In-place Local Write.
        Reads from neighbors' opposite index, collides, and writes locally to the original index.
        Includes implicit halfway bounce-back at solid-fluid interfaces.
        """
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.solid_mask[x, y, z] == 0:
                # 1. Pull populations from neighbors (or bounce-back locally if neighbor is solid)
                f_local = ti.Vector([0.0] * Q)

                for i in ti.static(range(Q)):
                    cx, cy, cz = c_tuple[i]
                    opp_i = opposite_tuple[i]

                    # Periodic wrapping on all axes
                    nb_x = (x - cx + self.nx) % self.nx
                    nb_y = (y - cy + self.ny) % self.ny
                    nb_z = (z - cz + self.nz) % self.nz

                    if self.solid_mask[nb_x, nb_y, nb_z] == 0:
                        # Neighbor is fluid, pull from neighbor's opposite slot
                        f_local[i] = self.f[opp_i, nb_x, nb_y, nb_z]
                    else:
                        # Neighbor is solid wall, perform halfway bounce-back from local cell
                        f_local[i] = self.f[i, x, y, z]

                # 2. Compute macroscopic density
                rho_val = 0.0
                for i in ti.static(range(Q)):
                    rho_val += f_local[i]
                self.rho[x, y, z] = rho_val

                # 3. Compute macroscopic momentum (raw velocity components)
                ux_val = 0.0
                uy_val = 0.0
                uz_val = 0.0
                for i in ti.static(range(Q)):
                    cx, cy, cz = c_tuple[i]
                    ux_val += f_local[i] * cx
                    uy_val += f_local[i] * cy
                    uz_val += f_local[i] * cz

                ux = 0.0
                uy = 0.0
                uz = 0.0
                if rho_val > 1e-8:
                    ux = ux_val / rho_val
                    uy = uy_val / rho_val
                    uz = uz_val / rho_val

                # Store physical velocity (with 0.5 * force shift)
                self.u[0, x, y, z] = ux + 0.5 * force_x
                self.u[1, x, y, z] = uy + 0.5 * force_y
                self.u[2, x, y, z] = uz + 0.5 * force_z

                # 4. Compute force-shifted velocity for equilibrium calculation
                ux_eq = ux + self.tau * force_x
                uy_eq = uy + self.tau * force_y
                uz_eq = uz + self.tau * force_z
                u2 = ux_eq * ux_eq + uy_eq * uy_eq + uz_eq * uz_eq

                # 5. Perform collision & write in-place to original index
                for i in ti.static(range(Q)):
                    cx, cy, cz = c_tuple[i]
                    wi = w_tuple[i]

                    cu = cx * ux_eq + cy * uy_eq + cz * uz_eq
                    feq = wi * rho_val * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

                    self.f[i, x, y, z] = f_local[i] - (f_local[i] - feq) / self.tau
            else:
                # Solid wall macro fields
                self.rho[x, y, z] = 1.0
                self.u[0, x, y, z] = 0.0
                self.u[1, x, y, z] = 0.0
                self.u[2, x, y, z] = 0.0

    def get_macroscopic_velocity(self) -> np.ndarray:
        """
        Copy the physical velocity field from Taichi memory to a CPU NumPy array.
        Returns array of shape (Nx, Ny, Nz, 3) in physical units.
        """
        # The Taichi u field is shaped (3, Nx, Ny, Nz) internally for optimal SoA coalescing
        u_cpu = self.u.to_numpy()
        # Transpose back to (Nx, Ny, Nz, 3) for standard user-facing representation
        return np.transpose(u_cpu, (1, 2, 3, 0))

    def get_density(self) -> np.ndarray:
        """
        Copy the density field from Taichi memory to a CPU NumPy array.
        Returns array of shape (Nx, Ny, Nz).
        """
        return self.rho.to_numpy()
