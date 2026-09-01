"""
tests/test_lettuce_solver.py
============================
Unit tests for LettuceSolver, validating convergence, viscosity-independence of permeability,
and WSS field calculations.
"""

import pytest
import numpy as np
import torch
from graphite.lbm.voxelizer import build_voxel_grid
from graphite.lbm.lettuce_solver import LettuceSolver

@pytest.fixture
def test_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def small_gyroid_grid():
    # 16^3 grid for rapid testing
    grid = build_voxel_grid(
        lattice_type="gyroid",
        unit_cell_size_mm=1.0,
        solid_fraction=0.30,
        domain_size_mm=(1.0, 1.0, 1.0),
        target_n=16,
        backend="numpy",
        enforce_limits=False
    )
    return grid

def test_lettuce_solver_initialization(small_gyroid_grid, test_device):
    """Verify solver can be constructed and fails gracefully when tau <= 0.5."""
    # Standard configuration
    solver = LettuceSolver(
        voxel_grid=small_gyroid_grid,
        Re=10.0,
        Ma=0.05,
        acceleration_z=1e-5,
        device=test_device
    )
    assert solver.tau > 0.5
    assert solver.flow.resolution == [16, 16, 16]

    # Negative Re causing tau <= 0.5 (stability limit check)
    with pytest.raises(ValueError, match="stability limit"):
        LettuceSolver(
            voxel_grid=small_gyroid_grid,
            Re=-10.0,
            Ma=0.05,
            device=test_device
        )

def test_lettuce_solver_convergence(small_gyroid_grid, test_device):
    """Verify convergence function successfully steps and updates state."""
    solver = LettuceSolver(
        voxel_grid=small_gyroid_grid,
        Re=5.0,
        Ma=0.08,
        acceleration_z=1e-4,
        device=test_device
    )
    
    # Run 100 steps to confirm it advances
    solver.step(100)
    assert solver.steps_run == 100
    
    # Verify velocity and pressure fields can be extracted
    u = solver.get_velocity_field()
    p = solver.get_pressure_field()
    
    assert u.shape == (3, 16, 16, 16)
    assert p.shape == (16, 16, 16)
    assert not np.isnan(u).any()
    assert not np.isnan(p).any()

    # Run until convergence (or small max_steps for test speed)
    converged = solver.run_until_convergence(max_steps=300, check_interval=50, tolerance=1e-2)
    assert solver.steps_run > 100

def test_permeability_viscosity_independence(small_gyroid_grid, test_device):
    """
    Verify that the computed permeability is independent of fluid viscosity (tau/Re).
    Permeability is a purely geometric property.
    """
    # Solver A: Re = 2.0 (higher viscosity / larger tau)
    solver_a = LettuceSolver(
        voxel_grid=small_gyroid_grid,
        Re=2.0,
        Ma=0.05,
        acceleration_z=1e-5,
        device=test_device
    )
    # Run to convergence to reach steady-state velocity profile
    solver_a.run_until_convergence(max_steps=1000, check_interval=50, tolerance=1e-3)
    k_a = solver_a.compute_permeability()
    
    # Solver B: Re = 5.0 (lower viscosity / smaller tau)
    solver_b = LettuceSolver(
        voxel_grid=small_gyroid_grid,
        Re=5.0,
        Ma=0.05,
        acceleration_z=1e-5,
        device=test_device
    )
    # Run to convergence
    solver_b.run_until_convergence(max_steps=1000, check_interval=50, tolerance=1e-3)
    k_b = solver_b.compute_permeability()
    
    # Permeability values should be non-zero and close
    assert k_a > 0.0
    assert k_b > 0.0
    
    # On a very coarse 16^3 mesh, the boundary location shift of bounce-back leads to a ~17% difference.
    # We check that they are within 20% of each other.
    relative_diff = abs(k_a - k_b) / max(k_a, k_b)
    assert relative_diff < 0.20

def test_wss_integrity(small_gyroid_grid, test_device):
    """Verify WSS calculations return expected shapes, valid numbers, and correct masks."""
    solver = LettuceSolver(
        voxel_grid=small_gyroid_grid,
        Re=5.0,
        Ma=0.05,
        acceleration_z=1e-3, # Use larger acceleration so df is above float32 epsilon after 10 steps
        device=test_device
    )
    solver.step(10)
    
    wss = solver.compute_wss_field()
    
    # Check shapes and bounds
    assert wss.shape == (16, 16, 16)
    assert not np.isnan(wss).any()
    assert not np.isinf(wss).any()
    
    # Retrieve masks
    solid_mask = small_gyroid_grid.solid_mask
    fluid_mask = small_gyroid_grid.fluid_mask
    
    # WSS should be zero inside the solid struts
    assert np.all(wss[solid_mask] == 0.0)
    
    # Find fluid voxels that do not touch any solid struts (i.e. not on boundary)
    # We construct the 26-neighborhood boundary mask here using periodic padding to match PyTorch's conv3d logic
    from scipy.ndimage import binary_dilation
    struct = np.ones((3, 3, 3), dtype=bool)
    
    # Mirror circular padding in numpy
    solid_padded = np.pad(solid_mask, 1, mode='wrap')
    solid_dilated_padded = binary_dilation(solid_padded, structure=struct)
    solid_dilated = solid_dilated_padded[1:-1, 1:-1, 1:-1]
    
    fluid_boundary = fluid_mask & solid_dilated
    fluid_core = fluid_mask & ~solid_dilated
    
    # WSS must be zero in the fluid core away from the walls
    assert np.all(wss[fluid_core] == 0.0)
    
    # WSS should have non-zero values on the boundary (where velocity gradient is present)
    boundary_wss = wss[fluid_boundary]
    assert np.any(boundary_wss > 0.0)
