"""
graphite.lbm.vocal_run — Vocal orchestration (grid → solve → metrics → plot).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from graphite.lbm.lettuce_solver import LettuceSolver
from graphite.lbm.voxelizer import (
    VoxelGrid,
    build_voxel_grid,
    pad_flow_chamber_mask,
    voxelize_stl_to_grid,
    _sample_abs_tpms_phase_distribution,
    _tau_from_solid_fraction_quantile,
)
from graphite.math.tpms import evaluate_tpms

BoundaryStyle = Literal["periodic", "flow_chamber", "flow_chamber_reverse"]
GeometryKind = Literal["stl", "tpms", "graded-tpms"]
WarmStartMode = Literal["none", "cache", "analytic", "auto"]

FLOW_CHAMBER_PAD_XY = 1
FLOW_CHAMBER_PAD_Z = 3
DEFAULT_COMPARISON_DPI = 300


@dataclass
class VocalRunConfig:
  geometry: GeometryKind = "stl"
  stl_path: str | Path | None = None
  lattice_type: str = "gyroid"
  unit_cell_size_mm: float = 1.0
  solid_fraction: float = 0.30
  domain_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)
  sf_inlet: float = 0.20
  sf_outlet: float = 0.60
  target_n: int = 128
  apply_flow_chamber_pad: bool = True
  boundary_style: BoundaryStyle = "flow_chamber"
  re: float = 5.0
  ma: float = 0.05
  acceleration_z: float = 1e-4
  device: str = "cuda"
  steps: int = 500
  converge: bool = False
  max_steps: int = 15000
  check_interval: int = 500
  tolerance: float = 1e-4
  min_steps: int = 0
  quiver_stride: int = 4
  plot_title: str | None = None
  output_png: Path | None = None
  metrics_json: Path | None = None
  plot: bool = True
  cache_dir: Path | None = None
  use_cache: bool = True
  force_rerun: bool = False
  warm_start: WarmStartMode = "auto"
  # Optional explicit warm-start overrides (used by coarse→fine workflows).
  warm_start_cache_dir: Path | None = None
  warm_start_flip_uz: bool = False
  warm_start_u_pu: object | None = None
  warm_start_p_pu: object | None = None
  warm_start_source: str | None = None


@dataclass
class VocalMetrics:
  converged: bool
  steps_run: int
  pressure_drop_pa: float
  permeability_mm2: float
  wss_mean_pa: float
  wss_median_pa: float
  wss_max_pa: float
  grid_shape: tuple[int, int, int]
  solid_fraction: float
  boundary_style: str
  geometry: str
  lattice_type: str
  notes: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def resolve_device(requested: str = "cuda") -> str:
  if requested == "cuda" and torch.cuda.is_available():
    return "cuda"
  return "cpu"


def build_graded_tpms_grid(
  *,
  lattice_type: str = "split-p",
  domain_size_mm: tuple[float, float, float] = (4.0, 4.0, 4.0),
  unit_cell_size_mm: float = 1.0,
  target_n: int = 128,
  sf_inlet: float = 0.20,
  sf_outlet: float = 0.60,
  apply_flow_chamber_pad: bool = True,
) -> VoxelGrid:
  f_dist = _sample_abs_tpms_phase_distribution(lattice_type, samples_per_axis=96)
  tau_inlet = _tau_from_solid_fraction_quantile(f_dist, sf_inlet)
  tau_outlet = _tau_from_solid_fraction_quantile(f_dist, sf_outlet)

  x = np.linspace(0, domain_size_mm[0], target_n, endpoint=False)
  y = np.linspace(0, domain_size_mm[1], target_n, endpoint=False)
  z = np.linspace(0, domain_size_mm[2], target_n, endpoint=False)
  X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

  k = 2.0 * np.pi / unit_cell_size_mm
  tpms_field = evaluate_tpms(lattice_type, k, X, Y, Z)
  z_frac = Z / domain_size_mm[2]
  tau_field = tau_inlet + z_frac * (tau_outlet - tau_inlet)
  solid_mask = np.abs(tpms_field) <= tau_field

  if apply_flow_chamber_pad:
    solid_mask = pad_flow_chamber_mask(solid_mask)

  nx, ny, nz = solid_mask.shape
  dx = domain_size_mm[0] / target_n
  return VoxelGrid(
    solid_mask=solid_mask,
    fluid_mask=~solid_mask,
    X=np.zeros_like(solid_mask, dtype=np.float32),
    Y=np.zeros_like(solid_mask, dtype=np.float32),
    Z=np.zeros_like(solid_mask, dtype=np.float32),
    tpms_field=np.zeros_like(solid_mask, dtype=np.float32),
    tau=0.0,
    nx=nx,
    ny=ny,
    nz=nz,
    voxel_size_mm=dx,
    domain_size_mm=(nx * dx, ny * dx, nz * dx),
    lattice_type=f"{lattice_type}_graded",
    unit_cell_size_mm=unit_cell_size_mm,
    solid_fraction_target=(sf_inlet + sf_outlet) / 2.0,
    solid_fraction_actual=float(np.mean(solid_mask)),
    backend="lettuce-cuda",
    vram_estimate_gb=0.5,
    ram_estimate_gb=2.0,
    notes=[f"Z-graded SF {sf_inlet:.2f} → {sf_outlet:.2f}"],
  )


def build_grid(config: VocalRunConfig) -> VoxelGrid:
  if config.geometry == "stl":
    if not config.stl_path:
      raise ValueError("geometry=stl requires stl_path")
    return voxelize_stl_to_grid(
      str(config.stl_path),
      target_n=config.target_n,
      apply_flow_chamber_pad=config.apply_flow_chamber_pad,
    )
  if config.geometry == "tpms":
    return build_voxel_grid(
      lattice_type=config.lattice_type,
      unit_cell_size_mm=config.unit_cell_size_mm,
      solid_fraction=config.solid_fraction,
      domain_size_mm=config.domain_size_mm,
      target_n=config.target_n,
      backend="numpy",
      enforce_limits=False,
    )
  if config.geometry == "graded-tpms":
    return build_graded_tpms_grid(
      lattice_type=config.lattice_type,
      domain_size_mm=config.domain_size_mm,
      unit_cell_size_mm=config.unit_cell_size_mm,
      target_n=config.target_n,
      sf_inlet=config.sf_inlet,
      sf_outlet=config.sf_outlet,
      apply_flow_chamber_pad=config.apply_flow_chamber_pad,
    )
  raise ValueError(f"Unknown geometry: {config.geometry}")


def run_solver(grid: VoxelGrid, config: VocalRunConfig) -> tuple[LettuceSolver, bool]:
    from graphite.lbm.vocal_warmstart import resolve_warm_start

    device = resolve_device(config.device)
    accel = (
        0.0
        if config.boundary_style in ("flow_chamber", "flow_chamber_reverse")
        else config.acceleration_z
    )

    warm = resolve_warm_start(config, grid, mode=config.warm_start)
    if warm.source:
        print(f"Warm start mode: {warm.mode} ({warm.source})", flush=True)
    elif warm.mode != "none":
        print(f"Warm start mode: {warm.mode}", flush=True)

    solver = LettuceSolver(
        voxel_grid=grid,
        Re=config.re,
        Ma=config.ma,
        acceleration_z=accel,
        device=device,
        boundary_style=config.boundary_style,
        initial_p_pu=warm.p_pu,
        initial_u_pu=warm.u_pu,
        warm_start_f_path=warm.f_path,
    )
    converged = False
    if config.converge:
        converged = solver.run_until_convergence(
            max_steps=config.max_steps,
            check_interval=config.check_interval,
            tolerance=config.tolerance,
            min_steps=config.min_steps,
        )
    else:
        solver.step(config.steps)
    return solver, converged


def collect_metrics(
  grid: VoxelGrid,
  solver: LettuceSolver,
  config: VocalRunConfig,
  *,
  converged: bool | None = None,
) -> VocalMetrics:
  p_field = solver.get_pressure_field()
  p_in = float(np.mean(p_field[:, :, 0]))
  p_out = float(np.mean(p_field[:, :, -1]))
  dp = p_in - p_out
  k_mm2 = float(solver.compute_permeability())
  wss_field = solver.compute_wss_field()
  wss_active = wss_field[wss_field > 1e-12]

  if len(wss_active) > 0:
    wss_mean = float(np.mean(wss_active))
    wss_median = float(np.median(wss_active))
    wss_max = float(np.max(wss_active))
  else:
    wss_mean = wss_median = wss_max = 0.0

  if converged is None:
    converged = config.converge

  return VocalMetrics(
    converged=bool(converged),
    steps_run=solver.steps_run,
    pressure_drop_pa=dp,
    permeability_mm2=k_mm2,
    wss_mean_pa=wss_mean,
    wss_median_pa=wss_median,
    wss_max_pa=wss_max,
    grid_shape=(grid.nx, grid.ny, grid.nz),
    solid_fraction=float(np.mean(grid.solid_mask)),
    boundary_style=config.boundary_style,
    geometry=config.geometry,
    lattice_type=grid.lattice_type,
  )


def plot_xz_velocity_slice(
  grid: VoxelGrid,
  flow,
  *,
  output_path: Path,
  title: str = "Vocal steady-state flow (XZ slice)",
  quiver_stride: int = 4,
  dpi: int = 150,
) -> Path:
  _, _, speed_sl, u_x_sl, u_z_sl, mask_sl, _ = extract_unit_cube_xz_slice(grid, flow)

  fig, ax = plt.subplots(figsize=(8, 10))
  _draw_xz_flow_panel(
    ax,
    speed_sl,
    mask_sl,
    u_x_sl,
    u_z_sl,
    title=title,
    quiver_stride=quiver_stride,
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)
  return output_path


def print_metrics(metrics: VocalMetrics) -> None:
    print("\n--- Vocal steady-state metrics ---")
    print(f"Converged:      {metrics.converged}")
    print(f"Steps:          {metrics.steps_run}")
    print(f"Grid:           {metrics.grid_shape[0]}x{metrics.grid_shape[1]}x{metrics.grid_shape[2]}")
    print(f"Solid fraction: {metrics.solid_fraction:.3f}")
    print(f"Pressure drop:  {metrics.pressure_drop_pa:.4f} Pa")
    print(f"Permeability:   {metrics.permeability_mm2:.6e} mm^2")
    print(f"WSS mean:       {metrics.wss_mean_pa:.4e} Pa")
    print(f"WSS median:     {metrics.wss_median_pa:.4e} Pa")
    print(f"WSS max:        {metrics.wss_max_pa:.4e} Pa")
    print("----------------------------------")


def run_vocal(config: VocalRunConfig) -> tuple[VoxelGrid, object, VocalMetrics]:
    from graphite.lbm.vocal_cache import cache_path_for, load_cache, save_cache

    cache_dir = cache_path_for(config, config.cache_dir)
    if config.use_cache and not config.force_rerun:
        cached = load_cache(cache_dir, config)
        if cached is not None:
            grid, flow, metrics = cached
            print(f"Cache hit: {cache_dir}")
            print_metrics(metrics)
            if config.metrics_json:
                config.metrics_json.parent.mkdir(parents=True, exist_ok=True)
                config.metrics_json.write_text(
                    json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
                )
                print(f"Metrics JSON: {config.metrics_json}")
            if config.plot and config.output_png:
                title = config.plot_title or f"Vocal — {config.geometry} ({config.boundary_style})"
                plot_xz_velocity_slice(
                    grid,
                    flow,
                    output_path=config.output_png,
                    title=title,
                    quiver_stride=config.quiver_stride,
                )
                print(f"Plot: {config.output_png}")
            return grid, flow, metrics

    grid = build_grid(config)
    print(f"Voxel grid: {grid.nx}x{grid.ny}x{grid.nz}  solid={grid.solid_fraction_actual:.3f}")

    solver, converged = run_solver(grid, config)
    if config.converge:
        if converged:
            print(f"Converged after {solver.steps_run} steps.")
        else:
            print(f"Reached {solver.steps_run} steps (convergence not confirmed).")
    else:
        print(f"Completed {solver.steps_run} steps.")

    metrics = collect_metrics(grid, solver, config, converged=converged)
    print_metrics(metrics)

    if config.metrics_json:
        config.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        config.metrics_json.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
        print(f"Metrics JSON: {config.metrics_json}")

    if config.plot and config.output_png:
        title = config.plot_title or f"Vocal — {config.geometry} ({config.boundary_style})"
        plot_xz_velocity_slice(
            grid,
            solver,
            output_path=config.output_png,
            title=title,
            quiver_stride=config.quiver_stride,
        )
        print(f"Plot: {config.output_png}")

    if config.use_cache:
        velocity = solver.get_velocity_field()
        pressure = solver.get_pressure_field()
        saved = save_cache(
            cache_dir, config, grid, velocity, metrics, flow=solver.flow, pressure=pressure
        )
        print(f"Cache saved: {saved}")

    return grid, solver, metrics


def extract_unit_cube_xz_slice(
    grid: VoxelGrid,
    flow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Return cropped lattice-interior XZ mid-Y slice (padding stripped).

    Returns (x_mm, z_mm, speed, u_x, u_z, solid_mask, peak_speed_m_s).
    """
    u_np = flow.get_velocity_field()
    speed = np.sqrt(u_np[0] ** 2 + u_np[1] ** 2 + u_np[2] ** 2)
    solid = grid.solid_mask
    nx, ny, nz = solid.shape
    j_mid = ny // 2

    i0, i1 = FLOW_CHAMBER_PAD_XY, nx - FLOW_CHAMBER_PAD_XY
    k0, k1 = FLOW_CHAMBER_PAD_Z, nz - FLOW_CHAMBER_PAD_Z

    speed_sl = speed[i0:i1, j_mid, k0:k1]
    u_x_sl = u_np[0, i0:i1, j_mid, k0:k1]
    u_z_sl = u_np[2, i0:i1, j_mid, k0:k1]
    mask_sl = solid[i0:i1, j_mid, k0:k1]

    n_x = i1 - i0
    n_z = k1 - k0
    x_mm = (np.arange(n_x) + 0.5) / n_x
    z_mm = (np.arange(n_z) + 0.5) / n_z

    fluid = ~mask_sl
    peak = float(np.max(speed_sl[fluid])) if np.any(fluid) else 0.0
    return x_mm, z_mm, speed_sl, u_x_sl, u_z_sl, mask_sl, peak


def _draw_xz_flow_panel(
    ax,
    speed_slice: np.ndarray,
    mask_slice: np.ndarray,
    u_x_slice: np.ndarray,
    u_z_slice: np.ndarray,
    *,
    title: str,
    vmax: float | None = None,
    quiver_stride: int = 4,
    colorbar: bool = True,
) -> float:
    """Render one CFD-style XZ panel (viridis speed, dark walls, white quiver)."""
    n_x, n_z = speed_slice.shape
    extent = [0, n_x, 0, n_z]
    fluid = ~mask_slice
    peak = float(np.max(speed_slice[fluid])) if np.any(fluid) else 0.0
    if peak <= 0.0:
        ax.set_title(f"{title}\n(no fluid in slice)")
        ax.axis("off")
        return 0.0

    plot_vmax = vmax if vmax is not None else peak
    speed_masked = np.ma.masked_where(mask_slice, speed_slice)
    im = ax.imshow(
        speed_masked.T,
        cmap="viridis",
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=plot_vmax,
    )

    mask_rgba = np.zeros((n_x, n_z, 4))
    mask_rgba[mask_slice] = [0.15, 0.15, 0.15, 1.0]
    ax.imshow(mask_rgba.transpose((1, 0, 2)), origin="lower", extent=extent)

    x_coords = np.arange(0, n_x, 1)
    z_coords = np.arange(0, n_z, 1)
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords, indexing="ij")
    X_stride = X_grid[::quiver_stride, ::quiver_stride]
    Z_stride = Z_grid[::quiver_stride, ::quiver_stride]
    u_x_stride = u_x_slice[::quiver_stride, ::quiver_stride]
    u_z_stride = u_z_slice[::quiver_stride, ::quiver_stride]
    fluid_stride = ~mask_slice[::quiver_stride, ::quiver_stride]

    px = X_stride[fluid_stride] + 0.5
    pz = Z_stride[fluid_stride] + 0.5
    ax.quiver(
        px,
        pz,
        u_x_stride[fluid_stride],
        u_z_stride[fluid_stride],
        color="white",
        pivot="middle",
    )

    ax.set_facecolor("white")
    ax.set_xlim(0, n_x)
    ax.set_ylim(0, n_z)
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Z (voxels)")
    ax.set_title(f"{title}\nSlice peak |u| = {peak * 1000.0:.2f} mm/s", fontsize=10)

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Velocity magnitude (m/s)")

    return peak


def plot_flow_comparison(
    cases: tuple[dict, ...],
    *,
    output_path: Path,
    dpi: int = DEFAULT_COMPARISON_DPI,
    suptitle: str | None = None,
    quiver_stride: int = 4,
    shared_color_scale: bool = True,
) -> Path:
    """
    Side-by-side XZ mid-Y flow comparison (CFD style: viridis, dark walls, quiver).

    Each case dict must include ``label``, ``grid``, and ``flow``.
    """
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    slices: list[tuple] = []
    peaks: list[float] = []
    for case in cases:
        _, _, speed_sl, u_x_sl, u_z_sl, mask_sl, peak = extract_unit_cube_xz_slice(
            case["grid"], case["flow"]
        )
        slices.append((speed_sl, mask_sl, u_x_sl, u_z_sl))
        peaks.append(peak)

    vmax = max(peaks) if shared_color_scale and peaks else None

    for ax, case, (speed_sl, mask_sl, u_x_sl, u_z_sl) in zip(
        axes, cases, slices, strict=True
    ):
        _draw_xz_flow_panel(
            ax,
            speed_sl,
            mask_sl,
            u_x_sl,
            u_z_sl,
            title=case["label"],
            vmax=vmax,
            quiver_stride=quiver_stride,
            colorbar=True,
        )

    if suptitle is None:
        suptitle = "Steady-state flow - Vertical cross-section"
    fig.suptitle(suptitle, fontsize=12, y=1.02)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def extract_unit_cube_xz_wss_slice(
    grid: VoxelGrid,
    wss_field: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Cropped XZ mid-Y WSS slice plus velocity components for streamlines."""
    solid = grid.solid_mask
    nx, ny, nz = solid.shape
    j_mid = ny // 2

    i0, i1 = FLOW_CHAMBER_PAD_XY, nx - FLOW_CHAMBER_PAD_XY
    k0, k1 = FLOW_CHAMBER_PAD_Z, nz - FLOW_CHAMBER_PAD_Z

    wss_sl = wss_field[i0:i1, j_mid, k0:k1]
    u_x_sl = velocity[0, i0:i1, j_mid, k0:k1]
    u_z_sl = velocity[2, i0:i1, j_mid, k0:k1]
    mask_sl = solid[i0:i1, j_mid, k0:k1]

    active = wss_sl[~mask_sl]
    peak = float(np.max(active)) if active.size > 0 else 0.0
    return wss_sl, u_x_sl, u_z_sl, mask_sl, peak


def smooth_interface_slice_for_display(
    field: np.ndarray,
    solid_mask: np.ndarray,
    *,
    sigma: float,
    support_threshold: float = 0.08,
) -> np.ndarray:
    """
    Mask-aware normalized Gaussian blur for sparse interface fields.

  Visualization only — does not change cached metrics or the underlying LBM
  solve. Spreads each non-zero sample over neighboring fluid voxels without
  bleeding into solid cells.
    """
    if sigma <= 0.0:
        return field

    from scipy.ndimage import gaussian_filter

    fluid = ~solid_mask
    active = (field > 0.0) & fluid
    if not np.any(active):
        return field

    values = np.where(active, field, 0.0).astype(np.float64)
    weight = active.astype(np.float64)
    num = gaussian_filter(values, sigma=sigma, mode="nearest")
    den = gaussian_filter(weight, sigma=sigma, mode="nearest")
    smoothed = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)
    smoothed = np.where(solid_mask | (den < support_threshold), 0.0, smoothed)
    return smoothed.astype(np.float32, copy=False)


def upsample_slice_for_display(
    field: np.ndarray,
    solid_mask: np.ndarray,
    *,
    factor: int,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsample a sparse 2D interface slice for display-only colormap rendering.

    The solid mask is nearest-neighbor upsampled so walls stay sharp; scalar
    fields use bilinear (order=1) or bicubic (order=3) interpolation. Values
    are cleared inside solid cells after resampling.
    """
    if factor <= 1:
        return field, solid_mask

    from scipy.ndimage import zoom

    factors = (int(factor), int(factor))
    mask_up = zoom(solid_mask.astype(np.float32), factors, order=0) >= 0.5
    fluid = ~solid_mask
    values = np.where((field > 0.0) & fluid, field, 0.0).astype(np.float32)
    field_up = zoom(values, factors, order=int(order))
    field_up = np.where(mask_up, 0.0, field_up)
    return np.clip(field_up, 0.0, None), mask_up


def _upsample_velocity_slice_for_display(
    u_x_slice: np.ndarray,
    u_z_slice: np.ndarray,
    solid_mask: np.ndarray,
    *,
    factor: int,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor <= 1:
        return u_x_slice, u_z_slice, solid_mask

    from scipy.ndimage import zoom

    factors = (int(factor), int(factor))
    mask_up = zoom(solid_mask.astype(np.float32), factors, order=0) >= 0.5
    u_x = np.where(solid_mask, 0.0, u_x_slice).astype(np.float32)
    u_z = np.where(solid_mask, 0.0, u_z_slice).astype(np.float32)
    u_x_up = zoom(u_x, factors, order=int(order))
    u_z_up = zoom(u_z, factors, order=int(order))
    u_x_up = np.where(mask_up, 0.0, u_x_up)
    u_z_up = np.where(mask_up, 0.0, u_z_up)
    return u_x_up, u_z_up, mask_up


def unit_cube_midplane_y_mm(grid: VoxelGrid) -> float:
    """Physical Y coordinate (mm) of the Vocal grid mid-plane slice."""
    j_mid = grid.ny // 2
    return (j_mid + 0.5) * float(grid.voxel_size_mm)


def stl_path_from_vocal_cache(cache_dir: Path) -> Path | None:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stl = manifest.get("fingerprint", {}).get("stl_path")
    return Path(stl) if stl else None


def rasterize_stl_xz_section(
    stl_path: str | Path,
    *,
    y_mm: float,
    n_x: int,
    n_z: int,
) -> np.ndarray:
    """
    Rasterize an STL Y-normal slice over the unit-cube XZ plane [0, 1]² mm.

    Returns a boolean solid mask with shape ``(n_x, n_z)`` suitable for smooth
    display geometry (vector slice, not Vocal voxels).
    """
    import trimesh
    from matplotlib.path import Path as MPath

    mesh = trimesh.load(str(stl_path), force="mesh")
    section = mesh.section(
        plane_origin=[0.0, float(y_mm), 0.0],
        plane_normal=[0.0, 1.0, 0.0],
    )
    if section is None:
        return np.zeros((int(n_x), int(n_z)), dtype=bool)

    path2d, to_3d = section.to_planar()
    to_2d = np.linalg.inv(to_3d)
    xs = (np.arange(n_x) + 0.5) / float(n_x)
    zs = (np.arange(n_z) + 0.5) / float(n_z)
    x_grid, z_grid = np.meshgrid(xs, zs, indexing="ij")
    pts3d = np.column_stack(
        [x_grid.ravel(), np.full(x_grid.size, float(y_mm)), z_grid.ravel()]
    )
    pts2d = trimesh.transformations.transform_points(pts3d, to_2d)[:, :2]

    inside = np.zeros(len(pts2d), dtype=bool)
    for entity in path2d.entities:
        verts = path2d.vertices[entity.points]
        if len(verts) >= 3 and getattr(entity, "closed", True):
            inside |= MPath(verts).contains_points(pts2d)
    return inside.reshape(int(n_x), int(n_z))


def stamp_wss_on_stl_section(
    wss_slice: np.ndarray,
    stl_solid: np.ndarray,
    *,
    display_factor: int = 4,
    display_order: int = 1,
    smooth_sigma: float = 0.0,
    interface_band_px: float = 2.0,
) -> np.ndarray:
    """
    Interpolate a coarse Vocal WSS slice onto a fine STL cross-section grid.

    WSS is upsampled from the simulation slice, lightly smoothed, then stamped
    into a narrow fluid band adjacent to the smooth STL wall. Visualization
    only — metrics remain tied to the Vocal voxel solve.
    """
    from scipy.ndimage import distance_transform_edt, zoom

    factor = max(1, int(display_factor))
    n_x = wss_slice.shape[0] * factor
    n_z = wss_slice.shape[1] * factor

    if stl_solid.shape != (n_x, n_z):
        stl_solid = (
            zoom(
                stl_solid.astype(np.float32),
                (n_x / stl_solid.shape[0], n_z / stl_solid.shape[1]),
                order=0,
            )
            >= 0.5
        )

    empty_mask = np.zeros_like(wss_slice, dtype=bool)
    wss_up, _ = upsample_slice_for_display(
        wss_slice,
        empty_mask,
        factor=factor,
        order=int(display_order),
    )
    if smooth_sigma > 0.0:
        wss_up = smooth_interface_slice_for_display(
            wss_up,
            stl_solid,
            sigma=float(smooth_sigma),
        )

    fluid = ~stl_solid
    dist_px = distance_transform_edt(fluid)
    band = (dist_px <= float(interface_band_px)) & fluid
    stamped = np.where(band, wss_up, 0.0)
    return np.where(stl_solid, 0.0, stamped).astype(np.float32, copy=False)


def _draw_xz_wss_streamline_panel(
    ax,
    wss_slice: np.ndarray,
    mask_slice: np.ndarray,
    u_x_slice: np.ndarray,
    u_z_slice: np.ndarray,
    *,
    title: str,
    vmax: float | None = None,
    density: float = 1.15,
    colorbar: bool = True,
    wss_smooth_sigma: float = 0.0,
    wss_display_factor: int = 1,
    wss_display_order: int = 1,
    stl_solid_mask: np.ndarray | None = None,
) -> float:
    """XZ slice: WSS magnitude (turbo) on black; only interface WSS bands are colored."""
    n_x_phys, n_z_phys = wss_slice.shape
    fluid = ~mask_slice
    active = wss_slice[fluid]
    raw_peak = float(np.max(active)) if active.size > 0 else 0.0
    if raw_peak <= 0.0:
        ax.set_title(f"{title}\n(no WSS in slice)")
        ax.axis("off")
        return 0.0

    display_factor = max(1, int(wss_display_factor))
    geometry_mask = mask_slice
    plot_field = wss_slice
    u_x_plot_src = u_x_slice
    u_z_plot_src = u_z_slice

    if stl_solid_mask is not None:
        plot_field = stamp_wss_on_stl_section(
            wss_slice,
            stl_solid_mask,
            display_factor=display_factor,
            display_order=int(wss_display_order),
            smooth_sigma=float(wss_smooth_sigma),
            interface_band_px=max(2.0, 0.75 * display_factor),
        )
        geometry_mask = stl_solid_mask
        if geometry_mask.shape != plot_field.shape:
            from scipy.ndimage import zoom

            geometry_mask = (
                zoom(
                    geometry_mask.astype(np.float32),
                    (
                        plot_field.shape[0] / geometry_mask.shape[0],
                        plot_field.shape[1] / geometry_mask.shape[1],
                    ),
                    order=0,
                )
                >= 0.5
            )
        u_x_plot_src, u_z_plot_src, _ = _upsample_velocity_slice_for_display(
            u_x_slice,
            u_z_slice,
            mask_slice,
            factor=display_factor,
            order=int(wss_display_order),
        )
    else:
        if wss_smooth_sigma > 0.0:
            plot_field = smooth_interface_slice_for_display(
                wss_slice,
                mask_slice,
                sigma=float(wss_smooth_sigma),
            )
        if display_factor > 1:
            plot_field, geometry_mask = upsample_slice_for_display(
                plot_field,
                mask_slice,
                factor=display_factor,
                order=int(wss_display_order),
            )
            u_x_plot_src, u_z_plot_src, _ = _upsample_velocity_slice_for_display(
                u_x_slice,
                u_z_slice,
                mask_slice,
                factor=display_factor,
                order=int(wss_display_order),
            )

    plot_vmax = vmax if vmax is not None else raw_peak
    wss_plot = np.ma.masked_where(geometry_mask | (plot_field <= 0.0), plot_field)
    cmap_obj = plt.get_cmap("turbo").copy()
    cmap_obj.set_bad(color="black")
    extent = [0, n_x_phys, 0, n_z_phys]
    ax.imshow(
        (wss_plot / (plot_vmax + 1e-15)).T,
        cmap=cmap_obj,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    n_x_up, n_z_up = plot_field.shape
    x = (np.arange(n_x_up) + 0.5) * (n_x_phys / n_x_up)
    z = (np.arange(n_z_up) + 0.5) * (n_z_phys / n_z_up)
    u_x_plot = np.ma.masked_where(geometry_mask, u_x_plot_src)
    u_z_plot = np.ma.masked_where(geometry_mask, u_z_plot_src)
    if np.any(~geometry_mask):
        ax.streamplot(
            x,
            z,
            u_x_plot.T,
            u_z_plot.T,
            color="white",
            linewidth=0.9,
            density=density,
            arrowsize=0.9,
        )

    ax.set_facecolor("black")
    ax.tick_params(colors="0.85")
    ax.xaxis.label.set_color("0.85")
    ax.yaxis.label.set_color("0.85")
    ax.set_xlim(0, n_x_phys)
    ax.set_ylim(0, n_z_phys)
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Z (voxels)")
    display_note = ""
    if stl_solid_mask is not None:
        display_note = f"  STL walls + Vocal stamp ×{display_factor}"
    elif display_factor > 1:
        display_note = f"  display ×{display_factor}"
    elif wss_smooth_sigma > 0.0:
        display_note = f"  display σ={wss_smooth_sigma:g} vox"
    ax.set_title(
        f"{title}\nSlice peak WSS = {raw_peak:.1f} Pa{display_note}",
        fontsize=10,
    )
    if colorbar:
        sm = ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=0.0, vmax=plot_vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="WSS (Pa)")
    return raw_peak


def plot_wss_comparison(
    cases: tuple[dict, ...],
    *,
    output_path: Path,
    dpi: int = DEFAULT_COMPARISON_DPI,
    suptitle: str | None = None,
    shared_color_scale: bool = False,
    streamline_density: float = 1.15,
    device: str = "cuda",
    wss_smooth_sigma: float = 0.0,
    wss_display_factor: int = 1,
    wss_display_order: int = 1,
    use_stl_cross_section: bool = False,
) -> Path:
    """
    Side-by-side XZ mid-Y WSS comparison with velocity streamlines.

    Each case dict must include ``label`` and ``vocal_cache`` (Path to cache dir).
    WSS is recomputed from cached ``f.pkl`` — no LBM re-simulation.
    """
    from graphite.lbm.vocal_cache import load_metrics_from_cache_dir, load_solver_from_cache_dir

    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    slices: list[tuple] = []
    peaks: list[float] = []
    stl_masks: list[np.ndarray | None] = []
    display_factor = max(1, int(wss_display_factor))
    for case in cases:
        cache_dir = Path(case["vocal_cache"])
        grid, solver = load_solver_from_cache_dir(cache_dir, device=device)
        metrics = load_metrics_from_cache_dir(cache_dir)
        wss_field = solver.compute_wss_field()
        velocity = solver.get_velocity_field()
        wss_sl, u_x_sl, u_z_sl, mask_sl, peak = extract_unit_cube_xz_wss_slice(
            grid, wss_field, velocity
        )
        steps = int(metrics.steps_run)
        conv = "converged" if metrics.converged else f"{steps} steps"
        case["panel_title"] = (
            f"{case['label']}\nVocal ({conv})  "
            f"WSS max = {metrics.wss_max_pa:.0f} Pa"
        )
        slices.append((wss_sl, mask_sl, u_x_sl, u_z_sl))
        peaks.append(peak)

        stl_mask = None
        if use_stl_cross_section:
            stl_path = case.get("stl_path") or stl_path_from_vocal_cache(cache_dir)
            if stl_path is None or not Path(stl_path).is_file():
                raise FileNotFoundError(
                    f"STL cross-section requested but no STL found for cache: {cache_dir}"
                )
            n_disp_x = wss_sl.shape[0] * display_factor
            n_disp_z = wss_sl.shape[1] * display_factor
            stl_mask = rasterize_stl_xz_section(
                stl_path,
                y_mm=unit_cube_midplane_y_mm(grid),
                n_x=n_disp_x,
                n_z=n_disp_z,
            )
        stl_masks.append(stl_mask)

    vmax = max(peaks) if shared_color_scale and peaks else None

    for ax, case, (wss_sl, mask_sl, u_x_sl, u_z_sl), stl_mask in zip(
        axes, cases, slices, stl_masks, strict=True
    ):
        _draw_xz_wss_streamline_panel(
            ax,
            wss_sl,
            mask_sl,
            u_x_sl,
            u_z_sl,
            title=case["panel_title"],
            vmax=vmax,
            density=streamline_density,
            colorbar=True,
            wss_smooth_sigma=wss_smooth_sigma,
            wss_display_factor=wss_display_factor,
            wss_display_order=wss_display_order,
            stl_solid_mask=stl_mask,
        )

    if suptitle is None:
        suptitle = "Vocal wall shear stress — XZ mid-plane (Re=5, flow chamber)"
    fig.suptitle(suptitle, fontsize=12, y=1.02)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def run_vocal_comparison(
    cases: tuple[dict, ...],
    *,
    base_config: VocalRunConfig,
    output_png: Path,
    summary_json: Path | None = None,
) -> tuple[tuple[VoxelGrid, LettuceSolver, VocalMetrics], ...]:
    """
  Run Vocal on each case (dict with ``label`` and optional ``stl_path``) and
  write an Aristo-style comparison figure.
    """
    results: list[tuple[VoxelGrid, LettuceSolver, VocalMetrics]] = []
    plot_cases: list[dict] = []

    for case in cases:
        label = case["label"]
        print(f"\n=== Vocal: {label} ===")
        cfg = VocalRunConfig(
            geometry=case.get("geometry", base_config.geometry),
            stl_path=case.get("stl_path", base_config.stl_path),
            lattice_type=case.get("lattice_type", base_config.lattice_type),
            target_n=case.get("target_n", base_config.target_n),
            apply_flow_chamber_pad=case.get(
                "apply_flow_chamber_pad", base_config.apply_flow_chamber_pad
            ),
            boundary_style=case.get("boundary_style", base_config.boundary_style),
            re=case.get("re", base_config.re),
            ma=case.get("ma", base_config.ma),
            acceleration_z=case.get("acceleration_z", base_config.acceleration_z),
            device=case.get("device", base_config.device),
            steps=case.get("steps", base_config.steps),
            converge=case.get("converge", base_config.converge),
            max_steps=case.get("max_steps", base_config.max_steps),
            check_interval=case.get("check_interval", base_config.check_interval),
            tolerance=case.get("tolerance", base_config.tolerance),
            plot=False,
            cache_dir=case.get("cache_dir", base_config.cache_dir),
            use_cache=case.get("use_cache", base_config.use_cache),
            force_rerun=case.get("force_rerun", base_config.force_rerun),
            warm_start=case.get("warm_start", base_config.warm_start),
        )
        metrics_path = case.get("metrics_json")
        if metrics_path:
            cfg.metrics_json = Path(metrics_path)

        grid, flow, metrics = run_vocal(cfg)
        results.append((grid, flow, metrics))
        plot_cases.append({"label": label, "grid": grid, "flow": flow})

    plot_flow_comparison(
        plot_cases,
        output_path=output_png,
        quiver_stride=base_config.quiver_stride,
    )
    print(f"Comparison figure: {output_png}")

    if summary_json is not None:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cases": [
                {
                    "label": cases[i]["label"],
                    "stl_path": str(cases[i].get("stl_path", "")),
                    "metrics": results[i][2].to_dict(),
                }
                for i in range(len(cases))
            ],
            "comparison_png": str(output_png),
        }
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return tuple(results)
