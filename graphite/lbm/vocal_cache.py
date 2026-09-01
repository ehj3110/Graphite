"""
graphite.lbm.vocal_cache — disk cache for Vocal runs (grid, velocity, metrics).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from graphite.lbm.voxelizer import VoxelGrid

CACHE_VERSION = 2
MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
VELOCITY_NAME = "velocity.npy"
PRESSURE_NAME = "pressure.npy"
SOLID_MASK_NAME = "solid_mask.npy"
F_DIST_NAME = "f.pkl"


class CachedFlowState:
    """Velocity holder compatible with Vocal plot helpers."""

    def __init__(self, velocity: np.ndarray):
        self._velocity = velocity

    def get_velocity_field(self) -> np.ndarray:
        return self._velocity


def default_cache_root() -> Path:
    return Path("outputs/vocal/cache")


def cache_slug(config) -> str:
    if config.geometry == "stl" and config.stl_path:
        stem = Path(config.stl_path).stem
    elif config.geometry == "graded-tpms":
        stem = f"{config.lattice_type}_graded"
    else:
        stem = f"{config.lattice_type}_{config.geometry}"

    if config.converge:
        run_id = f"conv_ms{config.max_steps}_tol{config.tolerance:g}"
    else:
        run_id = f"steps{config.steps}"

    return (
        f"{stem}_n{config.target_n}_{config.boundary_style}_"
        f"re{config.re:g}_ma{config.ma:g}_{run_id}"
    )


def cache_path_for(config, cache_root: Path | None = None) -> Path:
    root = cache_root or config.cache_dir or default_cache_root()
    return Path(root) / cache_slug(config)


def _source_fingerprint(config) -> dict:
    fp: dict = {
        "geometry": config.geometry,
        "target_n": config.target_n,
        "boundary_style": config.boundary_style,
        "re": config.re,
        "ma": config.ma,
        "converge": config.converge,
        "steps": config.steps,
        "max_steps": config.max_steps,
        "tolerance": config.tolerance,
        "apply_flow_chamber_pad": config.apply_flow_chamber_pad,
    }
    if config.geometry == "stl" and config.stl_path:
        stl = Path(config.stl_path)
        fp["stl_path"] = str(stl.resolve())
        if stl.is_file():
            stat = stl.stat()
            fp["stl_mtime"] = stat.st_mtime
            fp["stl_size"] = stat.st_size
    else:
        fp["lattice_type"] = config.lattice_type
        fp["domain_size_mm"] = list(config.domain_size_mm)
        fp["solid_fraction"] = config.solid_fraction
        fp["sf_inlet"] = config.sf_inlet
        fp["sf_outlet"] = config.sf_outlet
    return fp


def is_cache_valid(cache_dir: Path, config) -> bool:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("cache_version") != CACHE_VERSION:
        return False
    if manifest.get("fingerprint") != _source_fingerprint(config):
        return False
    return all((cache_dir / name).is_file() for name in (METRICS_NAME, VELOCITY_NAME, SOLID_MASK_NAME))


def _grid_from_mask(solid_mask: np.ndarray, meta: dict) -> VoxelGrid:
    nx, ny, nz = solid_mask.shape
    dx = float(meta["voxel_size_mm"])
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
        lattice_type=str(meta.get("lattice_type", "cached")),
        unit_cell_size_mm=float(meta.get("unit_cell_size_mm", 0.0)),
        solid_fraction_target=float(meta.get("solid_fraction_target", 0.0)),
        solid_fraction_actual=float(np.mean(solid_mask)),
        backend="lettuce-cuda",
        vram_estimate_gb=0.0,
        ram_estimate_gb=0.0,
        notes=["Restored from Vocal cache."],
    )


def save_cache(
    cache_dir: Path,
    config,
    grid: VoxelGrid,
    velocity: np.ndarray,
    metrics,
    *,
    flow=None,
    pressure: np.ndarray | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.save(cache_dir / VELOCITY_NAME, velocity.astype(np.float32, copy=False))
    np.save(cache_dir / SOLID_MASK_NAME, grid.solid_mask.astype(bool, copy=False))
    if pressure is not None:
        np.save(cache_dir / PRESSURE_NAME, pressure.astype(np.float32, copy=False))

    if flow is not None:
        flow.dump(str(cache_dir / F_DIST_NAME))

    metrics_path = cache_dir / METRICS_NAME
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")

    manifest = {
        "cache_version": CACHE_VERSION,
        "fingerprint": _source_fingerprint(config),
        "cache_slug": cache_slug(config),
        "grid_meta": {
            "voxel_size_mm": grid.voxel_size_mm,
            "lattice_type": grid.lattice_type,
            "unit_cell_size_mm": grid.unit_cell_size_mm,
            "solid_fraction_target": grid.solid_fraction_target,
        },
    }
    (cache_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache_dir


def load_cache(
    cache_dir: Path,
    config: object,
) -> tuple[VoxelGrid, CachedFlowState, object] | None:
    cache_dir = Path(cache_dir)
    if not is_cache_valid(cache_dir, config):  # type: ignore[arg-type]
        return None

    from graphite.lbm.vocal_run import VocalMetrics

    manifest = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    solid_mask = np.load(cache_dir / SOLID_MASK_NAME)
    velocity = np.load(cache_dir / VELOCITY_NAME)
    metrics_data = json.loads((cache_dir / METRICS_NAME).read_text(encoding="utf-8"))
    grid = _grid_from_mask(solid_mask, manifest["grid_meta"])
    metrics = VocalMetrics(
        converged=bool(metrics_data["converged"]),
        steps_run=int(metrics_data["steps_run"]),
        pressure_drop_pa=float(metrics_data["pressure_drop_pa"]),
        permeability_mm2=float(metrics_data["permeability_mm2"]),
        wss_mean_pa=float(metrics_data["wss_mean_pa"]),
        wss_median_pa=float(metrics_data["wss_median_pa"]),
        wss_max_pa=float(metrics_data["wss_max_pa"]),
        grid_shape=tuple(metrics_data["grid_shape"]),
        solid_fraction=float(metrics_data["solid_fraction"]),
        boundary_style=str(metrics_data["boundary_style"]),
        geometry=str(metrics_data["geometry"]),
        lattice_type=str(metrics_data["lattice_type"]),
        notes=list(metrics_data.get("notes", [])),
    )
    return grid, CachedFlowState(velocity), metrics


def load_grid_from_cache_dir(cache_dir: Path) -> VoxelGrid:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    solid_mask = np.load(cache_dir / SOLID_MASK_NAME)
    return _grid_from_mask(solid_mask, manifest["grid_meta"])


def load_metrics_from_cache_dir(cache_dir: Path):
    from graphite.lbm.vocal_run import VocalMetrics

    metrics_data = json.loads((Path(cache_dir) / METRICS_NAME).read_text(encoding="utf-8"))
    return VocalMetrics(
        converged=bool(metrics_data["converged"]),
        steps_run=int(metrics_data["steps_run"]),
        pressure_drop_pa=float(metrics_data["pressure_drop_pa"]),
        permeability_mm2=float(metrics_data["permeability_mm2"]),
        wss_mean_pa=float(metrics_data["wss_mean_pa"]),
        wss_median_pa=float(metrics_data["wss_median_pa"]),
        wss_max_pa=float(metrics_data["wss_max_pa"]),
        grid_shape=tuple(metrics_data["grid_shape"]),
        solid_fraction=float(metrics_data["solid_fraction"]),
        boundary_style=str(metrics_data["boundary_style"]),
        geometry=str(metrics_data["geometry"]),
        lattice_type=str(metrics_data["lattice_type"]),
        notes=list(metrics_data.get("notes", [])),
    )


def load_solver_from_cache_dir(cache_dir: Path, *, device: str = "cuda"):
    """
    Restore a Lettuce solver from cached ``f.pkl`` (no additional LBM steps).

    Requires ``f.pkl`` from a full Vocal run; velocity-only caches cannot recover WSS.
    """
    from graphite.lbm.lettuce_solver import LettuceSolver

    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    fp = manifest["fingerprint"]
    f_path = cache_dir / F_DIST_NAME
    if not f_path.is_file():
        raise FileNotFoundError(
            f"Missing {f_path.name} in {cache_dir} — WSS needs the saved LBM distribution."
        )
    grid = load_grid_from_cache_dir(cache_dir)
    solver = LettuceSolver(
        voxel_grid=grid,
        Re=float(fp["re"]),
        Ma=float(fp["ma"]),
        device=device,
        boundary_style=str(fp["boundary_style"]),
        warm_start_f_path=f_path,
    )
    return grid, solver
