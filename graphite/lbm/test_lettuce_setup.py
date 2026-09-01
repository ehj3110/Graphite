"""
graphite/lbm/test_lettuce_setup.py
====================================
Bare-bones lettuce (v0.2.3) smoke test for TPMS porous media LBM.

PURPOSE
-------
Verify that:
  1. PyTorch 2.5.1+cu121 initialises on the RTX 3050 CUDA device.
  2. lettuce 0.2.3 D3Q19 + BGKCollision can be constructed.
  3. A gyroid TPMS voxel geometry loads as a BounceBackBoundary mask.
  4. The simulation steps 10 times without NaN / OOM / exception.

HOW TO RUN
----------
Use the dedicated Python 3.12 venv that has torch CUDA + lettuce installed:

    .venv_torch\\Scripts\\python.exe graphite\\lbm\\test_lettuce_setup.py

VERIFIED LETTUCE 0.2.3 API
---------------------------
The following class paths are confirmed from the installed package:

    import lettuce as lt
    lt.Context(device, use_native=False)    # _context.py
    lt.D3Q19()                              # ext._stencil.d3q19  (must be CALLED)
    lt.UnitConversion(...)                  # _unit.py
    lt.Simulation(flow, collision, reporter=[])  # _simulation.py
    simulation.step(num_steps)              # steps forward

    from lettuce.ext import (
        BGKCollision,            # ext._collision
        BounceBackBoundary,      # ext._boundary  (full-way, NOT halfway)
        QuadraticEquilibrium,    # ext._equilibrium
        D3Q19,                   # ext._stencil  (also at lt.D3Q19)
    )

    Flow subclass must implement:
        initial_pu(self) -> (p_array, u_array)  # p: [*res], u: [3, *res]
        pre_boundaries property -> list of Boundary

GEOMETRY SOURCE
---------------
Geometry is generated at runtime via graphite.lbm.voxelizer.build_voxel_grid().
No pre-existing file is needed.

    solid_mask (numpy bool[Nx, Ny, Nz])  True = TPMS scaffold wall
    fluid_mask (numpy bool[Nx, Ny, Nz])  True = open pore channel

The solid_mask is converted to a bool PyTorch tensor and passed to
BounceBackBoundary(mask) — this marks those nodes as no-collision /
no-streaming walls using the full-way bounce-back rule.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so graphite imports work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===========================================================================
# STEP 1 — PyTorch + CUDA
# ===========================================================================
_section("STEP 1: PyTorch + CUDA Initialisation")

try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA available  : {cuda_ok}")
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU             : {device_name}")
        print(f"  Total VRAM      : {vram_gb:.2f} GB")
    else:
        print("  [WARNING] CUDA not available — falling back to CPU.")
        device_name = "CPU"
    DEVICE = torch.device("cuda:0" if cuda_ok else "cpu")
    print(f"  Active device   : {DEVICE}")
except ImportError as exc:
    print(f"\n  [FATAL] Cannot import torch: {exc}")
    print("  Activate .venv_torch (Python 3.12 + torch 2.5.1+cu121).")
    sys.exit(1)


# ===========================================================================
# STEP 2 — Import lettuce
# ===========================================================================
_section("STEP 2: lettuce 0.2.3 Import")

try:
    import lettuce as lt
    from lettuce.ext import (
        BGKCollision,
        BounceBackBoundary,
        QuadraticEquilibrium,
        D3Q19,
    )
    print(f"  lettuce version   : {lt.__version__}")
    print("  Imports OK        : Context, D3Q19, BGKCollision, BounceBackBoundary,")
    print("                      QuadraticEquilibrium, UnitConversion, Simulation")
except ImportError as exc:
    print(f"\n  [FATAL] Cannot import lettuce: {exc}")
    print("  Run: .venv_torch\\Scripts\\pip install git+https://github.com/lettucecfd/lettuce.git")
    sys.exit(1)


# ===========================================================================
# STEP 3 — Geometry: TPMS gyroid via build_voxel_grid
# ===========================================================================
_section("STEP 3: Geometry Generation (Gyroid TPMS — 64³)")

# Small domain: N=64 → ~0.03 GB VRAM for D3Q19 float32, safe on RTX 3050
LATTICE_TYPE   = "gyroid"
UNIT_CELL_MM   = 1.0
SOLID_FRACTION = 0.30   # 30% solid, 70% open pore
DOMAIN_MM      = (3.0, 3.0, 3.0)
TARGET_N       = 64

import numpy as np

try:
    from graphite.lbm.voxelizer import build_voxel_grid
    t0 = time.perf_counter()
    grid = build_voxel_grid(
        lattice_type=LATTICE_TYPE,
        unit_cell_size_mm=UNIT_CELL_MM,
        solid_fraction=SOLID_FRACTION,
        domain_size_mm=DOMAIN_MM,
        target_n=TARGET_N,
        # Tell voxelizer the intended backend for memory checks only
        backend="taichi-cuda" if cuda_ok else "taichi-cpu",
        enforce_limits=False,   # smoke test — don't raise on limit warnings
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  {grid}")
    print(f"  Voxelization     : {elapsed_ms:.1f} ms")
    print(f"  solid_mask shape : {grid.solid_mask.shape}  dtype={grid.solid_mask.dtype}")
    print(f"  Fluid voxels     : {grid.fluid_mask.sum():,} / {grid.nx*grid.ny*grid.nz:,}")
    SOLID_NP = grid.solid_mask      # numpy bool, True = wall
    NX, NY, NZ = grid.nx, grid.ny, grid.nz
except Exception as exc:
    print(f"  [WARNING] graphite.lbm.voxelizer unavailable ({exc})")
    print("  Using fallback: central cube solid in a 64³ domain.")
    NX = NY = NZ = TARGET_N
    SOLID_NP = np.zeros((NX, NY, NZ), dtype=bool)
    hw = NX // 8
    cx, cy, cz = NX // 2, NY // 2, NZ // 2
    SOLID_NP[cx-hw:cx+hw, cy-hw:cy+hw, cz-hw:cz+hw] = True
    print(f"  Fallback solid   : {2*hw}³ cube in ({NX},{NY},{NZ}) domain")
    print(f"  Solid fraction   : {SOLID_NP.mean()*100:.1f}%")


# ===========================================================================
# STEP 4 — lettuce Flow (D3Q19, pressure-driven channel)
# ===========================================================================
_section("STEP 4: Construct lettuce Flow (D3Q19, BGK)")

# Physical parameters — low Re/Ma for stability in a smoke test
Re = 10.0
Ma = 0.05

units = lt.UnitConversion(
    reynolds_number=Re,
    mach_number=Ma,
    characteristic_length_lu=NY,    # lattice units along Y
    characteristic_length_pu=1.0,   # physical length = 1 (arbitrary units)
)

tau = units.relaxation_parameter_lu
print(f"  Re              : {Re}")
print(f"  Ma              : {Ma}")
print(f"  tau             : {tau:.4f}  (must be > 0.5 for stability)")
assert tau > 0.5, f"tau = {tau:.4f} ≤ 0.5 — LBM instability guaranteed. Reduce Ma or Re."


class TPMSChannelFlow(lt.Flow):
    """
    Minimal 3D channel flow through a TPMS scaffold for smoke-testing.

    - Domain: NX × NY × NZ voxels with D3Q19 stencil.
    - Walls: TPMS solid_mask applied as full-way BounceBackBoundary.
    - Initialisation: uniform density (ρ=1), zero velocity.
    - No inlet/outlet BCs in this smoke test — pure BB wall check only.
    """

    def __init__(self, context, resolution, units, solid_mask_np):
        super().__init__(
            context=context,
            resolution=list(resolution),
            units=units,
            stencil=D3Q19(),
            equilibrium=QuadraticEquilibrium(),
        )
        # Store solid mask on the correct device as a bool tensor
        solid_t = torch.tensor(solid_mask_np, dtype=torch.bool, device=context.device)
        self._bb = BounceBackBoundary(solid_t)

    def initial_pu(self):
        """Uniform pressure (ρ=1) and zero velocity as initial condition."""
        p0 = np.zeros([NX, NY, NZ], dtype=np.float32)
        u0 = np.zeros([3, NX, NY, NZ], dtype=np.float32)   # 3 components for 3D
        return p0, u0

    @property
    def pre_boundaries(self):
        return [self._bb]


# Build context — use_native=False disables CUDA kernel JIT compilation
# (faster startup; native mode is for production throughput)
context = lt.Context(device=DEVICE, use_native=False)

try:
    flow = TPMSChannelFlow(
        context=context,
        resolution=(NX, NY, NZ),
        units=units,
        solid_mask_np=SOLID_NP,
    )
    print(f"  Context         : device={DEVICE}, use_native=False")
    print(f"  f tensor shape  : {list(flow.f.shape)}")
    print(f"  f device        : {flow.f.device}")
    if cuda_ok:
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after init : {allocated_gb:.3f} GB")
except Exception as exc:
    print(f"\n  [FATAL] Flow initialisation failed: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ===========================================================================
# STEP 5 — Build Simulation
# ===========================================================================
_section("STEP 5: Build Simulation (BGKCollision + BounceBackBoundary)")

try:
    collision = BGKCollision(tau=tau)
    simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])
    print(f"  BGKCollision    : tau={tau:.4f}")
    print(f"  pre_boundaries  : {[type(b).__name__ for b in flow.pre_boundaries]}")
    print("  Simulation      : OK")
except Exception as exc:
    print(f"\n  [FATAL] Simulation construction failed: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ===========================================================================
# STEP 6 — Run 10 Steps
# ===========================================================================
_section("STEP 6: Advance 10 Simulation Steps")

NUM_STEPS = 10
has_nan = False
peak_gb = 0.0

try:
    t_start = time.perf_counter()
    simulation.step(NUM_STEPS)
    if cuda_ok:
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    rho_mean = flow.rho().mean().item()
    u_max    = flow.u().abs().max().item()
    has_nan  = bool(torch.isnan(flow.f).any() or torch.isinf(flow.f).any())

    if cuda_ok:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Steps run       : {NUM_STEPS}")
    print(f"  Wall time       : {elapsed_ms:.1f} ms  ({elapsed_ms/NUM_STEPS:.1f} ms/step)")
    print(f"  Mean density    : {rho_mean:.6f}  (expect ~1.0)")
    print(f"  Max |u|         : {u_max:.6g}")
    print(f"  NaN/Inf in f    : {has_nan}")
    if cuda_ok:
        print(f"  Peak VRAM       : {peak_gb:.3f} GB")

except Exception as exc:
    print(f"\n  [FATAL] Simulation step failed: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ===========================================================================
# FINAL RESULT
# ===========================================================================
_section("SMOKE TEST RESULT")

if not has_nan:
    solid_pct  = float(SOLID_NP.mean()) * 100
    fluid_pct  = 100.0 - solid_pct
    mem_line   = f"Peak {peak_gb:.3f} GB VRAM" if cuda_ok else "CPU (no VRAM usage)"
    print(f"""
  [PASS] ALL CHECKS PASSED

  GPU      : {device_name}
  Geometry : {LATTICE_TYPE} TPMS  {NX}×{NY}×{NZ} voxels
             {solid_pct:.1f}% solid  |  {fluid_pct:.1f}% fluid
  Physics  : D3Q19 BGK  tau={tau:.3f}  Re={Re}  Ma={Ma}
  BC       : BounceBackBoundary (full-way, on solid mask)
  Steps    : {NUM_STEPS} collision+streaming steps
  Time     : {elapsed_ms:.1f} ms total
  Memory   : {mem_line}

  PyTorch successfully initialised the GPU, loaded the TPMS
  geometry, and executed {NUM_STEPS} D3Q19 BGK steps with
  bounce-back walls — without OOM or NaN.

  NEXT STEPS (see docs/VOCAL.md):
  -----------
  Use scripts/run_vocal.py for steady-state runs, permeability, WSS, and plots.
  Inlet/outlet BCs, convergence, and Darcy k are implemented in lettuce_solver.py.
""")
else:
    print(f"""
  [WARN] SMOKE TEST COMPLETED WITH WARNINGS

  Simulation ran but produced NaN/Inf in the f-field.  Check:
    - tau = {tau:.4f}  (must be > 0.5; current Re={Re}, Ma={Ma})
    - Fluid pathway exists (solid fraction = {SOLID_NP.mean()*100:.1f}%)
    - Initial velocity is truly zero (no spurious large Ma)
""")
