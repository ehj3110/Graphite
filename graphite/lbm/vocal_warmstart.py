"""
graphite.lbm.vocal_warmstart — initial-condition guesses for Vocal LBM runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from graphite.lbm.vocal_cache import (
    MANIFEST_NAME,
    VELOCITY_NAME,
    PRESSURE_NAME,
    F_DIST_NAME,
    SOLID_MASK_NAME,
    _source_fingerprint,
    cache_path_for,
    default_cache_root,
    is_cache_valid,
)
from graphite.lbm.voxelizer import VoxelGrid

WarmStartMode = Literal["none", "cache", "analytic", "auto"]

# Typical order-of-magnitude guess for 1 mm lattice permeability runs (m/s).
_ANALYTIC_UZ_FLUID_M_S = 0.005
# Match steady-state ΔP seen in Re=5 / Ma=0.05 cube runs (~4000 Pa).
_FLOW_CHAMBER_DP_PA = 4000.0


@dataclass
class WarmStartPayload:
    mode: str
    f_path: Path | None = None
    p_pu: np.ndarray | None = None
    u_pu: np.ndarray | None = None
    source: str | None = None

    @classmethod
    def cold(cls) -> WarmStartPayload:
        return cls(mode="none")


def _geometry_fingerprint(config) -> dict:
    fp = dict(_source_fingerprint(config))
    for key in ("converge", "steps", "max_steps", "tolerance"):
        fp.pop(key, None)
    return fp


def _fingerprints_match(manifest_fp: dict, geometry_fp: dict) -> bool:
    for key, value in geometry_fp.items():
        if manifest_fp.get(key) != value:
            return False
    return True


def find_geometry_cache(config, cache_root: Path | None = None) -> Path | None:
    """Find any cache entry for the same geometry/BCs (step count may differ)."""
    root = Path(cache_root or config.cache_dir or default_cache_root())
    if not root.is_dir():
        return None

    geometry_fp = _geometry_fingerprint(config)
    exact = cache_path_for(config, root)
    if is_cache_valid(exact, config):
        return exact

    candidates: list[tuple[float, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _fingerprints_match(manifest.get("fingerprint", {}), geometry_fp):
            continue
        if not (child / VELOCITY_NAME).is_file():
            continue
        mtime = manifest_path.stat().st_mtime
        candidates.append((mtime, child))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_analytic_initial_pu(
    grid: VoxelGrid,
    boundary_style: str,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    fluid = ~grid.solid_mask

    u_pu = np.zeros((3, nx, ny, nz), dtype=np.float32)
    uz = _ANALYTIC_UZ_FLUID_M_S
    if boundary_style == "flow_chamber_reverse":
        uz = -uz
    u_pu[2][fluid] = uz

    p_pu = np.zeros((nx, ny, nz), dtype=np.float32)
    if boundary_style in ("flow_chamber", "flow_chamber_reverse") and nz > 1:
        z_frac = np.arange(nz, dtype=np.float32) / float(nz - 1)
        # Higher pressure at the inlet end (broadcast along X/Y).
        if boundary_style == "flow_chamber":
            p_pu[:] = (_FLOW_CHAMBER_DP_PA * (1.0 - z_frac))[None, None, :]
        else:
            p_pu[:] = (_FLOW_CHAMBER_DP_PA * z_frac)[None, None, :]

    return p_pu, u_pu


def upsample_velocity_field(
    u_coarse: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Trilinear-upsample a (3, nx, ny, nz) velocity field onto ``target_shape``.

    Used for coarse→fine warm starts (e.g. n=64 → n=128).
    """
    from scipy.ndimage import zoom

    if u_coarse.ndim != 4 or u_coarse.shape[0] != 3:
        raise ValueError(f"Expected velocity shape (3, nx, ny, nz); got {u_coarse.shape}")
    src = u_coarse.shape[1:]
    factors = tuple(t / s for t, s in zip(target_shape, src, strict=True))
    out = np.empty((3, *target_shape), dtype=np.float32)
    for c in range(3):
        out[c] = zoom(u_coarse[c].astype(np.float32, copy=False), factors, order=1)
    return out


def upsample_scalar_field(
    field_coarse: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Trilinear-upsample a (nx, ny, nz) scalar field."""
    from scipy.ndimage import zoom

    if field_coarse.ndim != 3:
        raise ValueError(f"Expected scalar shape (nx, ny, nz); got {field_coarse.shape}")
    src = field_coarse.shape
    factors = tuple(t / s for t, s in zip(target_shape, src, strict=True))
    return zoom(field_coarse.astype(np.float32, copy=False), factors, order=1)


def _load_pressure_array(cache_dir: Path, grid: VoxelGrid) -> np.ndarray | None:
    """Load pressure (Pa) from cache, upsampling or extracting from f.pkl if needed."""
    cache_dir = Path(cache_dir)
    pressure_path = cache_dir / PRESSURE_NAME
    target = (grid.nx, grid.ny, grid.nz)

    if pressure_path.is_file():
        pressure = np.load(pressure_path)
        if pressure.shape == target:
            return pressure.astype(np.float32, copy=True)
        if pressure.ndim == 3:
            return upsample_scalar_field(pressure, target)

    velocity_path = cache_dir / VELOCITY_NAME
    if (cache_dir / F_DIST_NAME).is_file() and velocity_path.is_file():
        from graphite.lbm.vocal_cache import load_solver_from_cache_dir

        _g, solver = load_solver_from_cache_dir(cache_dir, device="cpu")
        pressure = solver.get_pressure_field().astype(np.float32, copy=False)
        if pressure.shape == target:
            return pressure
        if pressure.ndim == 3:
            return upsample_scalar_field(pressure, target)
    return None


def upsample_cache_to_grid(
    cache_dir: Path,
    grid: VoxelGrid,
    boundary_style: str,
    *,
    flip_uz: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load coarse velocity from ``cache_dir`` and upsample onto ``grid``."""
    velocity_path = Path(cache_dir) / VELOCITY_NAME
    if not velocity_path.is_file():
        return None
    velocity = np.load(velocity_path)
    if velocity.ndim != 4 or velocity.shape[0] != 3:
        return None

    target = (grid.nx, grid.ny, grid.nz)
    if velocity.shape[1:] == target:
        u_pu = velocity.astype(np.float32, copy=True)
    else:
        u_pu = upsample_velocity_field(velocity, target)

    fluid = ~grid.solid_mask
    u_pu[:, ~fluid] = 0.0
    if flip_uz:
        u_pu[2] = -u_pu[2]

    pressure = _load_pressure_array(cache_dir, grid)
    if pressure is not None:
        if flip_uz:
            pressure = pressure[:, :, ::-1].astype(np.float32, copy=True)
        p_pu = pressure
    else:
        p_pu, _ = build_analytic_initial_pu(grid, boundary_style)
    return p_pu, u_pu


def _load_velocity_ic(
    cache_dir: Path,
    grid: VoxelGrid,
    boundary_style: str,
    *,
    flip_uz: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    velocity_path = cache_dir / VELOCITY_NAME
    if not velocity_path.is_file():
        return None
    velocity = np.load(velocity_path)
    if velocity.shape != (3, grid.nx, grid.ny, grid.nz):
        # Allow coarse→fine upsample when shapes differ.
        return upsample_cache_to_grid(
            cache_dir, grid, boundary_style, flip_uz=flip_uz
        )

    fluid = ~grid.solid_mask
    u_pu = velocity.astype(np.float32, copy=True)
    u_pu[:, ~fluid] = 0.0
    if flip_uz:
        u_pu[2] = -u_pu[2]

    pressure = _load_pressure_array(cache_dir, grid)
    if pressure is not None:
        if flip_uz:
            pressure = pressure[:, :, ::-1].astype(np.float32, copy=True)
        p_pu = pressure
    elif boundary_style in ("flow_chamber", "flow_chamber_reverse"):
        p_pu, _ = build_analytic_initial_pu(grid, boundary_style)
    else:
        p_pu = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float32)
    return p_pu, u_pu


def resolve_warm_start(
    config,
    grid: VoxelGrid,
    *,
    mode: WarmStartMode = "auto",
) -> WarmStartPayload:
    if mode == "none":
        return WarmStartPayload.cold()

    # Explicit override fields (used by coarse→fine scripts).
    override_u = getattr(config, "warm_start_u_pu", None)
    override_p = getattr(config, "warm_start_p_pu", None)
    if override_u is not None:
        return WarmStartPayload(
            mode="override_velocity",
            p_pu=override_p,
            u_pu=override_u,
            source=getattr(config, "warm_start_source", "override"),
        )

    explicit_cache = getattr(config, "warm_start_cache_dir", None)
    flip_uz = bool(getattr(config, "warm_start_flip_uz", False))
    if explicit_cache is not None:
        cache_dir = Path(explicit_cache)
        f_path = cache_dir / F_DIST_NAME
        mask_path = cache_dir / SOLID_MASK_NAME
        # Prefer f.pkl only when grid shape matches (cannot upsample f).
        if f_path.is_file() and mask_path.is_file() and not flip_uz:
            mask = np.load(mask_path)
            if mask.shape == (grid.nx, grid.ny, grid.nz):
                return WarmStartPayload(
                    mode="cache_f",
                    f_path=f_path,
                    source=str(cache_dir),
                )
        ic = _load_velocity_ic(
            cache_dir, grid, config.boundary_style, flip_uz=flip_uz
        )
        if ic is not None:
            p_pu, u_pu = ic
            src_shape = tuple(np.load(cache_dir / VELOCITY_NAME).shape[1:])
            suffix = "_flipUz" if flip_uz else ""
            mode_name = (
                "cache_velocity"
                if src_shape == (grid.nx, grid.ny, grid.nz)
                else f"cache_velocity_upsample{suffix}"
            )
            return WarmStartPayload(
                mode=mode_name,
                p_pu=p_pu,
                u_pu=u_pu,
                source=str(cache_dir),
            )

    cache_dir: Path | None = None
    if mode in ("cache", "auto"):
        cache_dir = find_geometry_cache(config)

    if cache_dir is not None:
        f_path = cache_dir / F_DIST_NAME
        if f_path.is_file():
            return WarmStartPayload(
                mode="cache_f",
                f_path=f_path,
                source=str(cache_dir),
            )
        ic = _load_velocity_ic(cache_dir, grid, config.boundary_style)
        if ic is not None:
            p_pu, u_pu = ic
            return WarmStartPayload(
                mode="cache_velocity",
                p_pu=p_pu,
                u_pu=u_pu,
                source=str(cache_dir),
            )
        if mode == "cache":
            print(f"Warm start: no usable cache at {cache_dir}, falling back to cold start.")

    if mode in ("analytic", "auto"):
        p_pu, u_pu = build_analytic_initial_pu(grid, config.boundary_style)
        return WarmStartPayload(mode="analytic", p_pu=p_pu, u_pu=u_pu)

    return WarmStartPayload.cold()
