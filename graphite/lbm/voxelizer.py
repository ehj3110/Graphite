"""
graphite.lbm.voxelizer — Step 1 of the LBM Pipeline
=====================================================

Generates a 3D binary solid/fluid voxel grid directly from TPMS implicit
equations, with no STL generation or FVM meshing.

Design decisions
----------------
* Reuses ``graphite.math.tpms.evaluate_tpms`` and
  ``graphite.math.tpms.evaluate_tpms_phase`` verbatim — zero code duplication.
* Reuses ``graphite.implicit.calibration.sample_abs_tpms_phase_distribution``
  and ``tau_from_solid_fraction_quantile`` to map a target solid volume
  fraction to the correct iso-threshold τ, exactly as the existing implicit
  engine does.
* The hex-grid axis construction (``_symmetric_axis_knots``) mirrors the logic
  in ``graphite.explicit.hex_scaffold_module`` but operates on scalar linspace
  axes rather than 8-corner element arrays — so we get aligned, symmetric
  voxel grids without touching Gmsh or scikit-fem.
* All safety limits are calibrated for NVIDIA Mobile GTX 3050 (≈4 GB VRAM)
  with a 2.5 GB headroom ceiling, and for 32 GB system RAM.

Resolution quick-reference (D3Q19, float32, AA-streaming single-copy)
----------------------------------------------------------------------
  N=128  ->  ~0.24 GB VRAM   (very safe GPU)
  N=192  ->  ~0.82 GB VRAM   (comfortable GPU)
  N=256  ->  ~1.95 GB VRAM   (maximum recommended GPU on this hardware)
  N=320  ->  ~3.80 GB VRAM   (exceeds 2.5 GB ceiling — CPU only)
  N=320  ->  ~6.6  GB RAM    (fine on 32 GB system RAM)

Usage
-----
>>> from graphite.lbm.voxelizer import VoxelGrid, build_voxel_grid
>>> grid = build_voxel_grid(
...     lattice_type="gyroid",
...     unit_cell_size_mm=1.0,
...     solid_fraction=0.30,
...     domain_size_mm=(3.0, 3.0, 3.0),
...     voxel_resolution_mm=None,   # auto from target_n
...     target_n=128,
...     backend="taichi-cuda",
... )
>>> print(grid)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Internal graphite imports
#
# NOTE: We import ONLY from graphite.math.tpms.  We do NOT import from the
# graphite.implicit package — that package's __init__.py eagerly loads
# meshing_backends (skimage) and calibration loads pore_metrics (trimesh),
# neither of which is installed in the LBM venv and neither is needed here.
#
# The two calibration helpers (sample_abs_tpms_phase_distribution and
# tau_from_solid_fraction_quantile) are inlined below verbatim from
# graphite/implicit/calibration.py so that voxelizer.py has ZERO external
# coupling beyond graphite.math.tpms and NumPy.
# ---------------------------------------------------------------------------
from graphite.math.tpms import evaluate_tpms, evaluate_tpms_phase


# ===========================================================================
# Inlined calibration helpers (source: graphite/implicit/calibration.py)
# These are verbatim copies — do not modify independently; sync with source.
# ===========================================================================


def _sample_abs_tpms_phase_distribution(
    lattice_type: str,
    *,
    samples_per_axis: int = 96,
) -> np.ndarray:
    """Sorted |F| samples on one reference period (U, V, W in [0, 2*pi)).

    Verbatim copy of
    ``graphite.implicit.calibration.sample_abs_tpms_phase_distribution``.
    """
    axis = np.linspace(0.0, 2.0 * np.pi, int(samples_per_axis), endpoint=False)
    U, V, W = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.abs(evaluate_tpms_phase(lattice_type, U, V, W)).ravel()
    values.sort()
    return values


def _tau_from_solid_fraction_quantile(
    abs_values_sorted: np.ndarray,
    target_solid_fraction,
):
    """tau so that |F| <= tau encloses the target solid-fraction quantile.

    Verbatim copy of
    ``graphite.implicit.calibration.tau_from_solid_fraction_quantile``.
    """
    sf = np.asarray(target_solid_fraction, dtype=np.float64)
    q = np.clip(sf, 0.0, 1.0)
    flat_q = q.ravel()
    n = int(len(abs_values_sorted))
    pos = flat_q * float(n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    w = pos - lo
    flat_tau = abs_values_sorted[lo] * (1.0 - w) + abs_values_sorted[hi] * w
    out = flat_tau.reshape(q.shape)
    return float(out) if np.ndim(target_solid_fraction) == 0 else out



# ===========================================================================
# Hardware-constrained resolution limits
# ===========================================================================

# GTX 3050 Mobile: ~4 GB VRAM.  Target: comfortable < 2.5 GB.
# D3Q19, float32, AA single-copy: f(N^3 x 19 x 4) + macro(N^3 x 4 x 4) + mask(N^3 x 1),
# x1.25 Taichi overhead.  At N=256 this is ~1.95 GB — within limit.
# At N=320 it would be ~3.80 GB — exceeds 2.5 GB limit.
_GPU_MAX_N: int = 256
_GPU_VRAM_LIMIT_GB: float = 2.5

# 32 GB system RAM.  NumPy needs 2x f-arrays (f + f_new).  Target: < 16 GB.
# At N=320 -> ~6.6 GB (safe). At N=512 -> ~27 GB (marginal).
_CPU_MAX_N: int = 384
_CPU_RAM_LIMIT_GB: float = 16.0

# Minimum sensible resolution: at least 8 voxels per unit-cell side.
_MIN_VOXELS_PER_UNIT_CELL: int = 8

# Supported TPMS type strings (mirrors graphite.math.tpms.evaluate_tpms).
_SUPPORTED_TYPES = frozenset(
    [
        "gyroid",
        "schwarz-p",
        "schwarz primitive",
        "schwarz",
        "diamond",
        "schwarz-diamond",
        "schwarz diamond",
        "schwarz-d",
        "neovius",
        "lidinoid",
        "split-p",
        "split_p",
    ]
)

BackendLiteral = Literal["taichi-cuda", "taichi-cpu", "numpy"]


# ===========================================================================
# Public result type
# ===========================================================================


@dataclass(frozen=True)
class VoxelGrid:
    """
    Immutable container for a binary TPMS voxel grid and its metadata.

    Attributes
    ----------
    solid_mask : ndarray, bool, shape (Nx, Ny, Nz)
        True where a voxel is solid (TPMS material). False -> fluid channel.
    fluid_mask : ndarray, bool, shape (Nx, Ny, Nz)
        Complement of solid_mask for convenience.
    X, Y, Z : ndarray, float32, shape (Nx, Ny, Nz)
        World-coordinate meshgrids (mm) for each voxel centre.
    tpms_field : ndarray, float32, shape (Nx, Ny, Nz)
        Raw evaluated TPMS scalar field F(x,y,z) before thresholding.
    tau : float
        Iso-threshold used: voxel is solid where |F| <= tau.
    nx, ny, nz : int
        Voxel counts along each axis.
    voxel_size_mm : float
        Isotropic voxel side length (mm).
    domain_size_mm : tuple of float
        Physical (Lx, Ly, Lz) extent of the grid (mm).
    lattice_type : str
        TPMS type string used.
    unit_cell_size_mm : float
        Period of the TPMS (mm).
    solid_fraction_target : float
        Target solid volume fraction requested.
    solid_fraction_actual : float
        Actual solid volume fraction achieved in the grid.
    backend : str
        Backend used for field evaluation.
    vram_estimate_gb : float
        Estimated D3Q19 float32 VRAM cost for this grid (single-copy AA).
    ram_estimate_gb : float
        Estimated D3Q19 float32 RAM cost for NumPy fallback (two-copy).
    notes : list of str
        Diagnostic messages from the build process.
    """

    solid_mask: np.ndarray
    fluid_mask: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    tpms_field: np.ndarray
    tau: float
    nx: int
    ny: int
    nz: int
    voxel_size_mm: float
    domain_size_mm: tuple
    lattice_type: str
    unit_cell_size_mm: float
    solid_fraction_target: float
    solid_fraction_actual: float
    backend: str
    vram_estimate_gb: float
    ram_estimate_gb: float
    notes: list = field(default_factory=list)

    def __str__(self) -> str:
        sf_err = abs(self.solid_fraction_actual - self.solid_fraction_target)
        return (
            f"VoxelGrid("
            f"{self.lattice_type} | "
            f"N=({self.nx},{self.ny},{self.nz}) | "
            f"dx={self.voxel_size_mm:.4f} mm | "
            f"SF_target={self.solid_fraction_target:.3f} | "
            f"SF_actual={self.solid_fraction_actual:.3f} [err={sf_err:.3f}] | "
            f"VRAM~{self.vram_estimate_gb:.3f} GB | "
            f"RAM~{self.ram_estimate_gb:.3f} GB"
            f")"
        )

    def porosity(self) -> float:
        """Return fluid (void) volume fraction = 1 - solid_fraction_actual."""
        return float(1.0 - self.solid_fraction_actual)

    def summary_dict(self) -> dict:
        """Return a JSON-serializable summary of key grid metadata."""
        return {
            "lattice_type": self.lattice_type,
            "unit_cell_size_mm": self.unit_cell_size_mm,
            "solid_fraction_target": self.solid_fraction_target,
            "solid_fraction_actual": self.solid_fraction_actual,
            "porosity": self.porosity(),
            "tau": self.tau,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "voxel_size_mm": self.voxel_size_mm,
            "domain_size_mm": list(self.domain_size_mm),
            "vram_estimate_gb": self.vram_estimate_gb,
            "ram_estimate_gb": self.ram_estimate_gb,
            "backend": self.backend,
            "notes": list(self.notes),
        }


# ===========================================================================
# Internal helpers — axis construction (mirrors hex_scaffold_module logic)
# ===========================================================================


def _symmetric_axis_knots(lo: float, hi: float, step: float) -> np.ndarray:
    """
    Build a 1-D linspace axis spanning [lo, hi] with isotropic step ``step``.

    Mirrors the logic of ``_symmetric_axis_knots`` in
    ``graphite.explicit.hex_scaffold_module`` but returns voxel *centres*
    rather than node coordinates.  The number of cells is
    ``ceil((hi - lo) / step)`` and the axis is centred on the domain midpoint.
    """
    extent = float(hi) - float(lo)
    n_cells = max(1, int(np.ceil(extent / float(step))))
    half_span = 0.5 * n_cells * float(step)
    center = 0.5 * (lo + hi)
    node_lo = center - half_span
    return np.linspace(node_lo + 0.5 * step, node_lo + (n_cells - 0.5) * step, n_cells)


def _estimate_vram_gb(nx: int, ny: int, nz: int) -> float:
    """D3Q19 float32 single-copy AA VRAM estimate with 1.25x Taichi overhead."""
    Q = 19
    n = nx * ny * nz
    f_bytes = n * Q * 4          # distribution functions
    macro_bytes = n * 4 * 4      # rho, ux, uy, uz
    mask_bytes = n * 1            # solid mask (uint8 in Taichi)
    return (f_bytes + macro_bytes + mask_bytes) * 1.25 / 1e9


def _estimate_ram_gb(nx: int, ny: int, nz: int) -> float:
    """D3Q19 float32 two-copy NumPy RAM estimate with 1.20x overhead."""
    Q = 19
    n = nx * ny * nz
    f_bytes = 2 * n * Q * 4      # f + f_new (two copies)
    macro_bytes = n * 4 * 4
    mask_bytes = n * 1
    return (f_bytes + macro_bytes + mask_bytes) * 1.20 / 1e9


# ===========================================================================
# Resolution safety checks
# ===========================================================================


def check_resolution_limits(
    nx: int,
    ny: int,
    nz: int,
    backend: BackendLiteral,
    *,
    raise_on_violation: bool = True,
) -> list[str]:
    """
    Validate that the requested grid fits within hardware-safe memory budgets.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions in voxels.
    backend : str
        ``"taichi-cuda"``, ``"taichi-cpu"``, or ``"numpy"``.
    raise_on_violation : bool
        If True, raise ``MemoryError`` on a budget violation.
        If False, return warnings instead (useful for the caller to decide).

    Returns
    -------
    list of str
        Any warning messages generated (empty if all limits pass).

    Raises
    ------
    MemoryError
        If ``raise_on_violation=True`` and a limit is exceeded.
    ValueError
        If grid dimensions are non-positive.
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Grid dimensions must be positive; got ({nx}, {ny}, {nz}).")

    msgs: list[str] = []
    n_max = max(nx, ny, nz)
    vram = _estimate_vram_gb(nx, ny, nz)
    ram = _estimate_ram_gb(nx, ny, nz)

    if backend == "taichi-cuda":
        if n_max > _GPU_MAX_N or vram > _GPU_VRAM_LIMIT_GB:
            msg = (
                f"Requested grid N_max={n_max} with estimated VRAM {vram:.3f} GB "
                f"exceeds GPU safe limit (N_max={_GPU_MAX_N}, ceiling={_GPU_VRAM_LIMIT_GB} GB) "
                f"for GTX 3050 Mobile. Switch to backend='taichi-cpu' for larger grids."
            )
            if raise_on_violation:
                raise MemoryError(msg)
            msgs.append(f"WARNING: {msg}")

    elif backend in ("taichi-cpu", "numpy"):
        if ram > _CPU_RAM_LIMIT_GB:
            msg = (
                f"Estimated RAM {ram:.3f} GB exceeds {_CPU_RAM_LIMIT_GB} GB safe ceiling "
                f"(32 GB system, 2x safety factor). N_max={n_max}. Reduce grid size."
            )
            if raise_on_violation:
                raise MemoryError(msg)
            msgs.append(f"WARNING: {msg}")
        if n_max > _CPU_MAX_N:
            msgs.append(
                f"WARNING: N_max={n_max} > CPU soft limit {_CPU_MAX_N}. "
                "Performance may degrade severely."
            )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Must be 'taichi-cuda', 'taichi-cpu', or 'numpy'."
        )

    return msgs


# ===========================================================================
# Core builder
# ===========================================================================


def build_voxel_grid(
    lattice_type: str,
    unit_cell_size_mm: float,
    solid_fraction: float,
    domain_size_mm: tuple,
    *,
    voxel_resolution_mm: float | None = None,
    target_n: int | None = 128,
    backend: BackendLiteral = "taichi-cuda",
    calibration_samples_per_axis: int = 96,
    enforce_limits: bool = True,
) -> VoxelGrid:
    """
    Build a 3D binary solid/fluid voxel grid from a TPMS implicit equation.

    The voxelizer pipeline is:

    1.  Compute iso-threshold tau from the target solid fraction via the same
        calibration quantile method used throughout the existing implicit
        engine (``graphite.implicit.calibration``).
    2.  Build a symmetric Cartesian coordinate meshgrid aligned to
        ``domain_size_mm`` using the same axis-knot strategy as
        ``graphite.explicit.hex_scaffold_module._symmetric_axis_knots``.
    3.  Evaluate the TPMS scalar field F(X, Y, Z) via
        ``graphite.math.tpms.evaluate_tpms`` (float32).
    4.  Threshold: solid_mask = (|F| <= tau).
    5.  Pack results into an immutable ``VoxelGrid`` with hardware-budget
        metadata attached.

    Parameters
    ----------
    lattice_type : str
        TPMS equation name. Supported: ``"gyroid"``, ``"schwarz-p"``,
        ``"schwarz primitive"``, ``"diamond"``, ``"schwarz-d"``,
        ``"schwarz-diamond"``, ``"neovius"``, ``"lidinoid"``,
        ``"split-p"`` / ``"split_p"``.
    unit_cell_size_mm : float
        Physical period of the TPMS unit cell (mm). Sets spatial frequency
        ``k = 2*pi / unit_cell_size_mm``.
    solid_fraction : float
        Target solid volume fraction in (0, 1). The iso-threshold tau is chosen
        so that the fraction of voxels in one reference period where |F| <= tau
        equals this value.
    domain_size_mm : tuple of float
        (Lx, Ly, Lz) physical extents of the simulation domain (mm).
    voxel_resolution_mm : float or None
        Isotropic voxel side length (mm). If None, ``target_n`` is used to
        derive the resolution: ``dx = min(domain_size_mm) / target_n``.
        Providing both raises ``ValueError``.
    target_n : int or None
        Target number of voxels along the shortest domain axis.  Ignored when
        ``voxel_resolution_mm`` is provided. Default 128.
    backend : str
        Intended downstream solver backend. Used only for memory-budget
        checking; voxelization itself is always done in NumPy.
        One of ``"taichi-cuda"``, ``"taichi-cpu"``, ``"numpy"``.
    calibration_samples_per_axis : int
        Samples-per-axis for the reference-period calibration grid used to
        derive tau. Default 96 (~884K samples, ~15 ms, accurate to < 0.1%).
    enforce_limits : bool
        If True (default), raise ``MemoryError`` when estimated VRAM / RAM
        exceeds the hardware-safe ceiling for the chosen backend.

    Returns
    -------
    VoxelGrid
        Immutable container with solid_mask, fluid_mask, coordinate meshgrids,
        raw field, tau, metadata, and hardware budget estimates.

    Raises
    ------
    ValueError
        Bad inputs (unsupported lattice type, impossible solid fraction, etc.).
    MemoryError
        Grid exceeds hardware-safe memory budget and ``enforce_limits=True``.

    Examples
    --------
    GPU run (N=128 gyroid, 30 % solid):

    >>> grid = build_voxel_grid(
    ...     "gyroid", unit_cell_size_mm=1.0, solid_fraction=0.30,
    ...     domain_size_mm=(3.0, 3.0, 3.0), target_n=128,
    ...     backend="taichi-cuda",
    ... )

    CPU run (N=256 split-p, 25 % solid, larger domain):

    >>> grid = build_voxel_grid(
    ...     "split-p", unit_cell_size_mm=0.8, solid_fraction=0.25,
    ...     domain_size_mm=(5.0, 5.0, 5.0), target_n=256,
    ...     backend="taichi-cpu",
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    lattice_type = str(lattice_type).strip()
    if lattice_type.lower() not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported lattice_type {lattice_type!r}. "
            f"Supported: {sorted(_SUPPORTED_TYPES)}"
        )

    if unit_cell_size_mm <= 0:
        raise ValueError(f"unit_cell_size_mm must be > 0; got {unit_cell_size_mm}.")

    if not (0.0 < solid_fraction < 1.0):
        raise ValueError(
            f"solid_fraction must be in (0, 1); got {solid_fraction}."
        )

    Lx, Ly, Lz = (float(v) for v in domain_size_mm)
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise ValueError(f"domain_size_mm must all be positive; got {domain_size_mm}.")

    if voxel_resolution_mm is not None and target_n is not None:
        raise ValueError(
            "Provide either voxel_resolution_mm or target_n, not both."
        )

    # ------------------------------------------------------------------
    # 2. Resolve voxel resolution
    # ------------------------------------------------------------------
    notes: list[str] = []

    if voxel_resolution_mm is not None:
        dx = float(voxel_resolution_mm)
        if dx <= 0:
            raise ValueError(f"voxel_resolution_mm must be > 0; got {dx}.")
    else:
        n = int(target_n) if target_n is not None else 128
        if n < 8:
            raise ValueError(f"target_n must be >= 8; got {n}.")
        dx = min(Lx, Ly, Lz) / float(n)

    # Warn if resolution is too coarse relative to the unit cell.
    voxels_per_cell = unit_cell_size_mm / dx
    if voxels_per_cell < _MIN_VOXELS_PER_UNIT_CELL:
        msg = (
            f"Only {voxels_per_cell:.1f} voxels span one unit cell "
            f"(unit_cell_size_mm={unit_cell_size_mm}, dx={dx:.4f} mm). "
            f"Recommend >= {_MIN_VOXELS_PER_UNIT_CELL} voxels/cell for "
            "accurate TPMS geometry. Consider reducing unit_cell_size_mm "
            "or increasing resolution."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        notes.append(f"LOW_RESOLUTION: {msg}")

    # ------------------------------------------------------------------
    # 3. Build coordinate axes (mirrors hex_scaffold_module axis logic)
    # ------------------------------------------------------------------
    # Domain origin at (0, 0, 0); extend to (Lx, Ly, Lz).
    x_axis = _symmetric_axis_knots(0.0, Lx, dx)
    y_axis = _symmetric_axis_knots(0.0, Ly, dx)
    z_axis = _symmetric_axis_knots(0.0, Lz, dx)

    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)

    # ------------------------------------------------------------------
    # 4. Hardware-budget check
    # ------------------------------------------------------------------
    limit_msgs = check_resolution_limits(
        nx, ny, nz, backend, raise_on_violation=enforce_limits
    )
    notes.extend(limit_msgs)
    for m in limit_msgs:
        warnings.warn(m, UserWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # 5. Calibrate tau from target solid fraction
    #    (exact same method as graphite.implicit.calibration)
    # ------------------------------------------------------------------
    abs_values_sorted = _sample_abs_tpms_phase_distribution(
        lattice_type=lattice_type.lower(),
        samples_per_axis=calibration_samples_per_axis,
    )
    tau = float(_tau_from_solid_fraction_quantile(abs_values_sorted, solid_fraction))
    notes.append(
        f"Calibrated tau={tau:.6f} for lattice={lattice_type!r}, "
        f"solid_fraction={solid_fraction:.4f} "
        f"(calibration grid: {calibration_samples_per_axis}^3 samples)."
    )

    # ------------------------------------------------------------------
    # 6. Build coordinate meshgrids (float32 to save memory)
    # ------------------------------------------------------------------
    X, Y, Z = np.meshgrid(
        x_axis.astype(np.float32),
        y_axis.astype(np.float32),
        z_axis.astype(np.float32),
        indexing="ij",
    )

    # ------------------------------------------------------------------
    # 7. Evaluate TPMS scalar field
    #    evaluate_tpms returns the unrectified F(X, Y, Z) with is_sheet=False
    #    (raw level-set, not abs). We apply abs here for consistent
    #    sheet-network thresholding: solid where |F| <= tau.
    # ------------------------------------------------------------------
    k = (2.0 * np.pi) / unit_cell_size_mm
    F_raw = evaluate_tpms(lattice_type.lower(), k, X, Y, Z)
    F_raw = np.asarray(F_raw, dtype=np.float32)

    # ------------------------------------------------------------------
    # 8. Threshold -> binary solid mask
    # ------------------------------------------------------------------
    solid_mask = np.abs(F_raw) <= tau   # True = solid
    fluid_mask = ~solid_mask            # True = fluid / pore channel

    # ------------------------------------------------------------------
    # 9. Compute actual solid fraction and report deviation
    # ------------------------------------------------------------------
    sf_actual = float(np.mean(solid_mask))
    sf_err = abs(sf_actual - solid_fraction)
    notes.append(
        f"Solid fraction: target={solid_fraction:.4f}, "
        f"actual={sf_actual:.4f}, error={sf_err:.4f}."
    )
    if sf_err > 0.03:
        msg = (
            f"Solid fraction error {sf_err:.4f} > 3% tolerance. "
            "Consider increasing calibration_samples_per_axis (e.g. 128) "
            "or using a finer voxel resolution."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        notes.append(f"SF_WARNING: {msg}")

    if not np.any(fluid_mask):
        msg = "No fluid voxels found — solid fraction is 100%. Check tau or solid_fraction."
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        notes.append(f"CRITICAL: {msg}")

    # ------------------------------------------------------------------
    # 10. Memory estimates for downstream solver
    # ------------------------------------------------------------------
    vram_gb = _estimate_vram_gb(nx, ny, nz)
    ram_gb = _estimate_ram_gb(nx, ny, nz)

    notes.append(
        f"Grid: ({nx}, {ny}, {nz}) voxels | "
        f"dx={dx:.4f} mm | "
        f"VRAM~{vram_gb:.3f} GB (GPU, AA) | "
        f"RAM~{ram_gb:.3f} GB (CPU, two-copy NumPy)."
    )

    return VoxelGrid(
        solid_mask=solid_mask,
        fluid_mask=fluid_mask,
        X=X,
        Y=Y,
        Z=Z,
        tpms_field=F_raw,
        tau=tau,
        nx=nx,
        ny=ny,
        nz=nz,
        voxel_size_mm=float(dx),
        domain_size_mm=(float(Lx), float(Ly), float(Lz)),
        lattice_type=lattice_type,
        unit_cell_size_mm=float(unit_cell_size_mm),
        solid_fraction_target=float(solid_fraction),
        solid_fraction_actual=sf_actual,
        backend=backend,
        vram_estimate_gb=vram_gb,
        ram_estimate_gb=ram_gb,
        notes=notes,
    )


# ===========================================================================
# Convenience wrappers for common hardware presets
# ===========================================================================


def build_voxel_grid_gpu(
    lattice_type: str,
    unit_cell_size_mm: float,
    solid_fraction: float,
    domain_size_mm: tuple,
    *,
    target_n: int = 128,
    **kwargs,
) -> VoxelGrid:
    """
    Convenience wrapper: GPU backend (taichi-cuda) with GTX 3050 safe defaults.

    Caps ``target_n`` at ``_GPU_MAX_N`` (256) and enforces the 2.5 GB VRAM
    ceiling.  Raises ``MemoryError`` if the requested grid exceeds limits.
    """
    if target_n > _GPU_MAX_N:
        raise MemoryError(
            f"target_n={target_n} exceeds GPU safe limit {_GPU_MAX_N} "
            f"for GTX 3050 Mobile (2.5 GB VRAM ceiling). "
            f"Use build_voxel_grid_cpu() for N > {_GPU_MAX_N}."
        )
    return build_voxel_grid(
        lattice_type=lattice_type,
        unit_cell_size_mm=unit_cell_size_mm,
        solid_fraction=solid_fraction,
        domain_size_mm=domain_size_mm,
        target_n=target_n,
        backend="taichi-cuda",
        enforce_limits=True,
        **kwargs,
    )


def build_voxel_grid_cpu(
    lattice_type: str,
    unit_cell_size_mm: float,
    solid_fraction: float,
    domain_size_mm: tuple,
    *,
    target_n: int = 192,
    **kwargs,
) -> VoxelGrid:
    """
    Convenience wrapper: CPU backend (taichi-cpu) using 32 GB system RAM.

    Allows grids up to ``_CPU_MAX_N`` (384) and enforces the 16 GB RAM ceiling.
    """
    return build_voxel_grid(
        lattice_type=lattice_type,
        unit_cell_size_mm=unit_cell_size_mm,
        solid_fraction=solid_fraction,
        domain_size_mm=domain_size_mm,
        target_n=target_n,
        backend="taichi-cpu",
        enforce_limits=True,
        **kwargs,
    )


# ===========================================================================
# Export / reload helpers
# ===========================================================================


def save_voxel_grid(grid: VoxelGrid, path: str) -> None:
    """
    Save a ``VoxelGrid`` to a compressed NumPy archive (``.npz``).

    Saves ``solid_mask``, ``tpms_field``, ``X``, ``Y``, ``Z``, and all
    scalar metadata fields.  Reload with ``load_voxel_grid()``.

    Parameters
    ----------
    grid : VoxelGrid
    path : str
        Output file path.  ``.npz`` extension will be added if absent.
    """
    np.savez_compressed(
        path,
        solid_mask=grid.solid_mask,
        fluid_mask=grid.fluid_mask,
        tpms_field=grid.tpms_field,
        X=grid.X,
        Y=grid.Y,
        Z=grid.Z,
        tau=np.float32(grid.tau),
        nx=np.int32(grid.nx),
        ny=np.int32(grid.ny),
        nz=np.int32(grid.nz),
        voxel_size_mm=np.float32(grid.voxel_size_mm),
        domain_size_mm=np.array(grid.domain_size_mm, dtype=np.float32),
        lattice_type=np.bytes_(grid.lattice_type),
        unit_cell_size_mm=np.float32(grid.unit_cell_size_mm),
        solid_fraction_target=np.float32(grid.solid_fraction_target),
        solid_fraction_actual=np.float32(grid.solid_fraction_actual),
        backend=np.bytes_(grid.backend),
        vram_estimate_gb=np.float32(grid.vram_estimate_gb),
        ram_estimate_gb=np.float32(grid.ram_estimate_gb),
    )


def load_voxel_grid(path: str) -> VoxelGrid:
    """
    Load a ``VoxelGrid`` previously saved with ``save_voxel_grid()``.

    Parameters
    ----------
    path : str
        Path to the ``.npz`` file.

    Returns
    -------
    VoxelGrid
    """
    data = np.load(path, allow_pickle=False)
    solid_mask = data["solid_mask"].astype(bool)
    return VoxelGrid(
        solid_mask=solid_mask,
        fluid_mask=~solid_mask,
        tpms_field=data["tpms_field"].astype(np.float32),
        X=data["X"].astype(np.float32),
        Y=data["Y"].astype(np.float32),
        Z=data["Z"].astype(np.float32),
        tau=float(data["tau"]),
        nx=int(data["nx"]),
        ny=int(data["ny"]),
        nz=int(data["nz"]),
        voxel_size_mm=float(data["voxel_size_mm"]),
        domain_size_mm=tuple(float(v) for v in data["domain_size_mm"]),
        lattice_type=str(data["lattice_type"]),
        unit_cell_size_mm=float(data["unit_cell_size_mm"]),
        solid_fraction_target=float(data["solid_fraction_target"]),
        solid_fraction_actual=float(data["solid_fraction_actual"]),
        backend=str(data["backend"]),
        vram_estimate_gb=float(data["vram_estimate_gb"]),
        ram_estimate_gb=float(data["ram_estimate_gb"]),
        notes=["Loaded from disk."],
    )

def pad_flow_chamber_mask(solid_mask: np.ndarray) -> np.ndarray:
    """
    Pad a base TPMS solid mask to create a virtual flow chamber.
    
    1. Adds a 1-voxel solid casing on the X and Y bounds to prevent flow escaping laterally.
    2. Adds a 3-voxel fluid buffer on the top and bottom (Z axis) to allow flow to enter
       homogeneously and equalize pressure before hitting the struts, preventing NaNs
       at the inlet/outlet boundaries.
       
    Parameters
    ----------
    solid_mask : np.ndarray
        3D boolean numpy array representing the unpadded TPMS solid mask.
        
    Returns
    -------
    np.ndarray
        Padded boolean mask.
    """
    # 1. Pad X/Y with solid (True)
    # np.pad takes pad_width=((before_1, after_1), (before_2, after_2), ...)
    mask = np.pad(solid_mask, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=True)
    
    # 2. Pad Z with fluid (False)
    mask = np.pad(mask, pad_width=((0, 0), (0, 0), (3, 3)), mode='constant', constant_values=False)
    
    return mask

def voxelize_stl_to_grid(stl_path: str, target_n: int = 128, apply_flow_chamber_pad: bool = True) -> VoxelGrid:
    """
    Load an STL file and voxelize it using trimesh.
    
    Parameters
    ----------
    stl_path : str
        Path to the STL file.
    target_n : int
        Target number of voxels along the longest axis. Default 128.
    apply_flow_chamber_pad : bool
        If True, applies the pad_flow_chamber_mask utility.
        
    Returns
    -------
    VoxelGrid
        The voxelized grid container ready for Lattice Boltzmann.
    """
    import trimesh
    
    print(f"Loading STL from {stl_path}...")
    mesh = trimesh.load(stl_path)
    
    pitch = mesh.extents.max() / target_n
    print(f"Voxelizing STL with pitch {pitch:.6f} mm...")
    
    # voxelized() creates a boundary voxelization, fill() fills the inside
    voxelized_mesh = mesh.voxelized(pitch=pitch)
    try:
        filled_mesh = voxelized_mesh.fill()
        solid_mask = filled_mesh.matrix
    except Exception as e:
        print(f"Warning: trimesh fill() failed ({e}), falling back to surface mask.")
        solid_mask = voxelized_mesh.matrix
        
    print(f"Base STL solid mask shape: {solid_mask.shape}")
    
    if apply_flow_chamber_pad:
        solid_mask = pad_flow_chamber_mask(solid_mask)
        print(f"Padded solid mask shape: {solid_mask.shape}")
        
    nx, ny, nz = solid_mask.shape
    dx = pitch
    
    # We create a VoxelGrid with dummy fields for X, Y, Z, tpms_field 
    # since this came from an explicit STL, not an implicit TPMS.
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
        lattice_type="STL_import",
        unit_cell_size_mm=0.0,
        solid_fraction_target=0.0,
        solid_fraction_actual=np.mean(solid_mask),
        backend="taichi-cuda",
        vram_estimate_gb=0.5,
        ram_estimate_gb=2.0
    )
