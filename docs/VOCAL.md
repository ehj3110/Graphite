# Vocal — LBM fluid simulation for lattice geometries

**Vocal** is Graphite’s lattice-Boltzmann fluid path for permeability and wall-shear-stress (WSS) studies on TPMS and STL lattice geometries. It uses **Lettuce** (D3Q19 BGK) on **PyTorch/CUDA** — not a traditional Navier–Stokes FEM/CFD stack.

**Code:** `graphite/lbm/`  
**Entry point:** `scripts/run_vocal.py`  
**Library orchestration:** `graphite/lbm/vocal_run.py` → `LettuceSolver`  
**Tests:** `tests/test_lettuce_solver.py`  
**Environment:** `.venv_torch` (PyTorch + Lettuce; separate from Aristo)

---

## What Vocal computes

| Output | Units | Method |
|--------|-------|--------|
| Steady-state velocity field | m/s | D3Q19 BGK + bounce-back on solids |
| Pressure field | Pa | Density–pressure mapping via Lettuce units |
| Darcy permeability **k** | mm² | Body-force or pressure-drop form of Darcy’s law |
| Wall shear stress (WSS) | Pa | Stress tensor from non-equilibrium distributions on fluid–solid interface |

Flow direction is **+Z** (bottom → top in standard voxel plots).

---

## Pipeline overview

```
Geometry source          Voxel grid              LBM solve              Post-process
─────────────────────────────────────────────────────────────────────────────────
TPMS implicit     ──►   build_voxel_grid()  ──►  LettuceSolver    ──►  k, ΔP, WSS
STL watertight    ──►   voxelize_stl_to_grid()     step / converge      XZ quiver PNG
Graded TPMS field ──►   build_graded_tpms_grid()
```

### Step 1 — Voxelization (`graphite/lbm/voxelizer.py`)

- **TPMS:** `build_voxel_grid(lattice_type, unit_cell_size_mm, solid_fraction, domain_size_mm, target_n)`
- **STL:** `voxelize_stl_to_grid(stl_path, target_n, apply_flow_chamber_pad=True)` via trimesh
- **Graded TPMS:** `build_graded_tpms_grid(...)` in `vocal_run.py` — Z-linear solid-fraction ramp

**Flow-chamber padding** (`pad_flow_chamber_mask`):

1. 1-voxel solid casing on X/Y faces (lateral confinement)
2. 3-voxel fluid buffer on Z inlet/outlet (pressure equalization before the lattice)

### Step 2 — Solver (`graphite/lbm/lettuce_solver.py`)

`LettuceSolver(voxel_grid, Re, Ma, device, boundary_style, acceleration_z)`

| `boundary_style` | Driving mechanism | Typical use |
|------------------|-------------------|-------------|
| `periodic` | Guo body-force along Z (`acceleration_z`) | Permeability on periodic domains; gyroid demos |
| `flow_chamber` | `EquilibriumOutletP` inlet/outlet | STL / graded studies with explicit ΔP |

Default physics: water-like density (1000 kg/m³), Reynolds and Mach set lattice viscosity (τ must stay **> 0.5**).

### Step 3 — Steady state

- `solver.step(n)` — fixed step count
- `solver.run_until_convergence(max_steps, check_interval, tolerance)` — tracks relative change in mean Z-velocity

### Step 4 — Metrics

```python
k_mm2 = solver.compute_permeability()
wss = solver.compute_wss_field()          # (NX, NY, NZ), Pa on interface voxels
u = solver.get_velocity_field()           # (3, NX, NY, NZ), m/s
p = solver.get_pressure_field()           # (NX, NY, NZ), Pa
```

---

## Environment setup

Vocal requires the **torch venv** (not the main Graphite venv used for Aristo/Gmsh):

```powershell
# From repo root — venv should already exist
.\.venv_torch\Scripts\python.exe -m pytest tests/test_lettuce_solver.py -q

# Smoke test (10 BGK steps on gyroid grid)
.\.venv_torch\Scripts\python.exe graphite\lbm\test_lettuce_setup.py
```

**Hardware notes** (from `voxelizer.py`):

| Resolution | ~VRAM (D3Q19 float32) | Notes |
|------------|----------------------|-------|
| 128³ | ~0.24 GB | Recommended default on 4 GB mobile GPUs |
| 192³ | ~0.82 GB | Comfortable |
| 256³ | ~1.95 GB | Upper practical GPU limit (GTX 3050 class) |

---

## Canonical CLI — `scripts/run_vocal.py`

Run from repo root with `.venv_torch`:

### STL lattice (flow chamber, convergence + metrics)

```powershell
.\.venv_torch\Scripts\python.exe scripts\run_vocal.py `
  --geometry stl `
  --stl experiments/implicit_to_volume/output/SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned.stl `
  --target-n 128 `
  --boundary-style flow_chamber `
  --converge `
  --output outputs/vocal/graded_cube_flow.png
```

### Uniform TPMS (periodic body-force demo)

```powershell
.\.venv_torch\Scripts\python.exe scripts\run_vocal.py `
  --geometry tpms `
  --lattice-type gyroid `
  --solid-fraction 0.30 `
  --domain-size-mm 2 2 2 `
  --target-n 64 `
  --boundary-style periodic `
  --steps 500 `
  --output outputs/vocal/gyroid_xz.png
```

### Z-graded implicit Split-P (flow chamber)

```powershell
.\.venv_torch\Scripts\python.exe scripts\run_vocal.py `
  --geometry graded-tpms `
  --lattice-type split-p `
  --domain-size-mm 4 4 4 `
  --sf-inlet 0.20 --sf-outlet 0.60 `
  --target-n 128 `
  --boundary-style flow_chamber `
  --converge `
  --output outputs/vocal/graded_splitp_flow.png
```

### Health check only (no simulation)

```powershell
.\.venv_torch\Scripts\python.exe scripts\run_vocal.py --health-check
```

Writes optional JSON metrics with `--metrics-json path.json`.

### Quick run with cache + warm start (recommended)

```powershell
# 1000 fixed steps, warm-start auto (cache f/velocity if present, else analytic guess)
.\.venv_torch\Scripts\python.exe scripts\run_vocal.py `
  --geometry stl `
  --stl experiments/implicit_to_volume/output/SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_cleaned.stl `
  --target-n 128 --steps 1000 --boundary-style flow_chamber `
  --warm-start auto `
  --output outputs/vocal/piecewise_flow.png `
  --metrics-json outputs/vocal/piecewise_metrics.json

# Re-run with fewer refinement steps from cached f distribution
.\.venv_torch\Scripts\python.exe scripts\run_vocal.py `
  --geometry stl --stl ... --steps 500 --warm-start auto --force-rerun --no-plot
```

| Flag | Role |
|------|------|
| `--warm-start auto` | Load cached `f.pkl` if available, else cached velocity, else analytic IC (default) |
| `--warm-start cache` | Cache only; cold start if missing |
| `--warm-start analytic` | Uniform +Z in fluid + inlet/outlet pressure ramp |
| `--warm-start none` | Cold start (u = 0) |
| `--force-rerun` | Ignore result cache but still warm-start from prior cache |
| `--no-cache` | Do not read/write result cache |

Result cache lives under `outputs/vocal/cache/{slug}/` (`velocity.npy`, `pressure.npy`, `f.pkl`, `metrics.json`).

Cross-resolution and direction-reversal warm-start (32→64→128 cascades, flip-Uz upflow seed): **[VOCAL_CASCADE_WARMSTART.md](VOCAL_CASCADE_WARMSTART.md)**.

### WSS plots from cache (no re-simulation)

Wall shear stress requires the full LBM distribution **`f.pkl`**, not `velocity.npy` alone. If `f.pkl` exists, WSS can be recomputed in seconds:

```powershell
.\.venv_torch\Scripts\python.exe scripts/plot_cube_1mm_vocal_wss_comparison.py
```

API: `load_solver_from_cache_dir()` in `vocal_cache.py` → `LettuceSolver.compute_wss_field()` → `plot_wss_comparison()` in `vocal_run.py`.

Three-lattice figure and cache slugs: **[CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md)**.

### Piecewise vs linear-graded 1 mm cube comparison

```powershell
# Quick preview (1000 fixed steps, both cases)
.\.venv_torch\Scripts\python.exe scripts\plot_vocal_flow_comparison.py --cases both --steps 1000

# Replot from cache only
.\.venv_torch\Scripts\python.exe scripts\plot_vocal_flow_comparison.py --cases both --steps 1000 --plot-only

# Toward steady state (see docs/VOCAL_CUBE_COMPARISON.md)
.\.venv_torch\Scripts\python.exe scripts\plot_vocal_flow_comparison.py `
  --cases both --converge --warm-start cache --max-steps 15000 --tolerance 1e-3
```

Status, metrics, and continuation options: **[VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md)**.

---

## Module layout

| File | Role |
|------|------|
| `voxelizer.py` | `VoxelGrid`, `build_voxel_grid`, `voxelize_stl_to_grid`, `pad_flow_chamber_mask` |
| `lettuce_solver.py` | `LettuceSolver`, `TPMSFlow` — Lettuce D3Q19 BGK, BCs, k, WSS |
| `vocal_run.py` | Orchestration: grid builders, run loop, metrics bundle, XZ slice plot |
| `vocal_cache.py` | Disk cache: velocity, full `f` distribution, metrics; `load_solver_from_cache_dir()` for WSS post-process |
| `vocal_warmstart.py` | Initial guesses: cache `f`, cached velocity + pressure upsample, analytic IC, flip-Uz |
| `d3q19.py` | Lattice constants (used by legacy Taichi path) |
| `lbm_solver.py` | Legacy Taichi BGK solver (not used by Vocal CLI) |
| `test_lettuce_setup.py` | GPU smoke test (10 steps) |

---

## Boundary-condition guidance

**Use `flow_chamber`** when:

- Geometry comes from an STL export
- You need a physical pressure drop (ΔP) for Darcy k
- Lateral walls should not be periodic

**Use `periodic`** when:

- Domain is a repeating unit cell (gyroid, diamond, etc.)
- Body-force drives flow; permeability uses k = U·ν/a

Always apply `pad_flow_chamber_mask` (default for STL import) when using `flow_chamber`.

---

## Visualization convention

XZ mid-Y slices plot:

- **X** horizontal, **Z** vertical (flow upward)
- Solid voxels: dark gray overlay
- Fluid speed: viridis heatmap
- White quiver arrows: local (u_x, u_z)

---

## Legacy scripts (deprecated)

These thin wrappers remain for backward compatibility; prefer `scripts/run_vocal.py`:

| Legacy script | Replacement |
|---------------|-------------|
| `scripts/visualize_stl_steady_state.py` | `run_vocal.py --geometry stl --converge` |
| `scripts/visualize_xz_slice.py` | `run_vocal.py --geometry tpms` |
| `scripts/visualize_graded_xz.py` | `run_vocal.py --geometry graded-tpms` |

---

## Known limits

- **Voxel resolution:** STL features smaller than one voxel pitch are lost; 128³ is the current default for 1 mm cubes.
- **Lettuce `no_streaming_mask`:** Patched in `LettuceSolver.__init__` (upstream bug workaround).
- **Coarse grids:** Permeability viscosity-independence is only ~20% on 16³ (see unit test); use 128³+ for comparisons.
- **Analytic flat caps:** FEA analytic-cap STLs are not yet the default Vocal inputs; align STL paths when running piecewise vs graded CFD comparisons.
- **No Streamlit UI:** Vocal is headless CLI only today.

---

## Planned next steps

1. **Converged cube comparison** — see [VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md) for current status, cached metrics @ 1000 steps, and continuation guidance.
2. **Analytic-cap STLs** — point Vocal at the same geometries used in Aristo FEA.
3. **Paired FEA/CFD summary** — converged k, ΔP, WSS alongside Aristo σ_vm for piecewise vs linear grading.

See also: [ARISTO.md](ARISTO.md) (structural FEA on the same lattice families), [VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md) (1 mm cube handoff).
