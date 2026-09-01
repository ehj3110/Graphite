"""
Vocal — Lattice Boltzmann fluid simulation for lattice geometries.

**Vocal** is the user-facing name for Graphite's LBM pipeline (permeability, WSS).
Implementation lives in ``graphite.lbm``; see ``docs/VOCAL.md``.

Modules
-------
voxelizer       : Step 1 — TPMS implicit or STL → binary solid/fluid voxel grid.
lettuce_solver  : Step 2–4 — Lettuce D3Q19 BGK solver, BCs, permeability, WSS.
vocal_run       : Orchestration — grid builders, run loop, metrics, XZ slice plots.
d3q19           : D3Q19 lattice constants (legacy Taichi path).
lbm_solver      : Legacy Taichi BGK solver (not used by Vocal CLI).
"""

from graphite.lbm.lettuce_solver import LettuceSolver
from graphite.lbm.vocal_run import VocalMetrics, VocalRunConfig, plot_flow_comparison, run_vocal, run_vocal_comparison
from graphite.lbm.voxelizer import VoxelGrid, build_voxel_grid, voxelize_stl_to_grid

__all__ = [
    "LettuceSolver",
    "VocalMetrics",
    "VocalRunConfig",
    "VoxelGrid",
    "build_voxel_grid",
    "plot_flow_comparison",
    "run_vocal",
    "run_vocal_comparison",
    "voxelize_stl_to_grid",
]
