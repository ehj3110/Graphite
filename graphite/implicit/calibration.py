"""
Graphite Implicit Engine - Calibration

This module handles the non-linear calibration of Triply Periodic Minimal 
Surface (TPMS) fields to achieve target geometric properties, specifically 
target pore sizes and solid volume fractions. It uses a Newton-Raphson-like 
iterative solver with Jacobian approximation, falling back to heuristic scaling 
when gradients vanish or misbehave. It supports both single-point calibration 
and Z-profile gradient calibration with interpolation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from graphite.implicit.density_control import tau_from_wall_thickness_mm
from graphite.implicit.pore_metrics import compute_max_inscribed_sphere_pore_size
from graphite.math.tpms import evaluate_tpms, evaluate_tpms_phase


@dataclass(frozen=True)
class CalibrationConfig:
    pore_tolerance_mm: float = 0.03
    solid_fraction_tolerance: float = 0.02
    max_iterations: int = 20
    damping: float = 0.7
    fd_relative_step_L: float = 0.03
    fd_absolute_step_tau: float = 0.02
    min_L_mm: float = 0.05
    max_L_mm: float = 50.0
    min_tau: float = 0.01
    max_tau: float = 2.0
    sample_resolution_mm: float = 0.02
    sample_cells: float = 2.0
    boundary_guard_cells: float = 0.5
    min_voxels_per_period: int = 28


@dataclass(frozen=True)
class CalibrationIteration:
    iteration: int
    L_mm: float
    tau: float
    pore_mis_mm: float
    solid_fraction_effective: float
    residual_pore_mm: float
    residual_solid_fraction: float


@dataclass(frozen=True)
class CalibrationPointResult:
    lattice_type: str
    target_pore_mm: float
    target_solid_fraction: float
    calibrated_L_mm: float
    calibrated_tau: float
    converged: bool
    iterations: int
    residual_pore_mm: float
    residual_solid_fraction: float
    measured_pore_mm: float
    measured_solid_fraction: float
    hit_bounds: bool
    history: list[CalibrationIteration] = field(default_factory=list)
    seed_source: str = "heuristic"
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationGradientResult:
    mode: str
    z_points_mm: list[float]
    target_pores_mm: list[float]
    target_solid_fractions: list[float]
    calibrated_L_mm: list[float]
    calibrated_tau: list[float]
    control_point_results: list[CalibrationPointResult]
    converged_all: bool
    notes: list[str] = field(default_factory=list)


def _default_seed_table_path() -> Path:
    return Path(__file__).with_name("calibration_seed_table.json")


def load_calibration_seed_table(seed_table_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(seed_table_path) if seed_table_path else _default_seed_table_path()
    if not path.exists():
        return {"entries": []}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "entries" not in data:
        return {"entries": []}
    if not isinstance(data["entries"], list):
        return {"entries": []}
    return data


def save_calibration_seed_table(
    seed_table: dict[str, Any],
    seed_table_path: str | Path | None = None,
) -> None:
    path = Path(seed_table_path) if seed_table_path else _default_seed_table_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(seed_table, fh, indent=2)


def update_seed_table_with_result(
    result: CalibrationPointResult,
    seed_table: dict[str, Any] | None = None,
    persist: bool = False,
    seed_table_path: str | Path | None = None,
) -> dict[str, Any]:
    table = seed_table if seed_table is not None else load_calibration_seed_table(seed_table_path)
    entries = table.setdefault("entries", [])
    entries.append(
        {
            "lattice_type": result.lattice_type,
            "target_pore_mm": result.target_pore_mm,
            "target_solid_fraction": result.target_solid_fraction,
            "L_mm": result.calibrated_L_mm,
            "tau": result.calibrated_tau,
        }
    )
    if persist:
        save_calibration_seed_table(table, seed_table_path=seed_table_path)
    return table


def sample_abs_tpms_phase_distribution(
    lattice_type: str,
    *,
    samples_per_axis: int = 96,
) -> np.ndarray:
    """Sorted |F| samples on one reference period (U,V,W in [0, 2π))."""
    axis = np.linspace(0.0, 2.0 * np.pi, int(samples_per_axis), endpoint=False)
    U, V, W = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.abs(evaluate_tpms_phase(lattice_type, U, V, W)).ravel()
    values.sort()
    return values


def tau_from_solid_fraction_quantile(
    abs_values_sorted: np.ndarray,
    target_solid_fraction: float | np.ndarray,
) -> float | np.ndarray:
    """τ so that ``|F| <= τ`` encloses the target solid-fraction quantile."""
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


def calibrate_tau_at_fixed_period(
    lattice_type: str,
    period_mm: float,
    target_solid_fraction: float,
    *,
    config: CalibrationConfig | None = None,
) -> CalibrationPointResult:
    """
    Hold unit-cell period ``L = period_mm`` fixed; solve only τ for solid fraction.

    Use this when the design specifies **unit-cell size** (wavelength), not
    maximum inscribed pore diameter (``calibrate_tpms_point``).
    """
    cfg = config or CalibrationConfig()
    if period_mm <= 0:
        raise ValueError("period_mm must be > 0")
    if not (0.0 < target_solid_fraction < 1.0):
        raise ValueError("target_solid_fraction must be within (0, 1)")

    abs_vals = sample_abs_tpms_phase_distribution(lattice_type)
    tau = float(tau_from_solid_fraction_quantile(abs_vals, target_solid_fraction))
    pore_mis, sf_eff = _sample_tpms_metrics(lattice_type, float(period_mm), tau, cfg)
    return CalibrationPointResult(
        lattice_type=lattice_type,
        target_pore_mm=float(period_mm),
        target_solid_fraction=float(target_solid_fraction),
        calibrated_L_mm=float(period_mm),
        calibrated_tau=tau,
        converged=True,
        iterations=1,
        residual_pore_mm=float(pore_mis - period_mm),
        residual_solid_fraction=float(sf_eff - target_solid_fraction),
        measured_pore_mm=float(pore_mis),
        measured_solid_fraction=float(sf_eff),
        hit_bounds=False,
        history=[],
        seed_source="fixed_period_tau_quantile",
        notes=[
            "Period held fixed; tau from |F| quantile. "
            "target_pore_mm stores period_mm for seed-table compatibility."
        ],
    )


def _nearest_seed_guess(
    lattice_type: str,
    target_pore_mm: float,
    target_solid_fraction: float,
    seed_table: dict[str, Any],
) -> tuple[float | None, float | None]:
    entries = seed_table.get("entries", [])
    candidates: list[tuple[float, float, float]] = []
    for entry in entries:
        if str(entry.get("lattice_type", "")).lower() != lattice_type.lower():
            continue
        L = entry.get("L_mm")
        tau = entry.get("tau")
        p = entry.get("target_pore_mm")
        sf = entry.get("target_solid_fraction")
        if any(v is None for v in [L, tau, p, sf]):
            continue
        score = abs(float(p) - target_pore_mm) + 2.0 * abs(float(sf) - target_solid_fraction)
        candidates.append((score, float(L), float(tau)))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    _, L_best, tau_best = candidates[0]
    return L_best, tau_best


def _heuristic_seed(target_pore_mm: float, target_solid_fraction: float) -> tuple[float, float]:
    # Use pore as first-order period guess and linearized SF->tau estimate.
    tau = target_solid_fraction
    tau = float(np.clip(tau, 0.05, 1.2))
    return max(target_pore_mm, 0.05), tau


def _sample_tpms_metrics(
    lattice_type: str,
    L_mm: float,
    tau: float,
    cfg: CalibrationConfig,
) -> tuple[float, float]:
    domain_mm = max(cfg.sample_cells * L_mm, L_mm)
    n = max(int(np.ceil(domain_mm / cfg.sample_resolution_mm)) + 1, cfg.min_voxels_per_period)
    # Hard cap to avoid OOM when L grows during Newton steps.
    max_n = 96
    if n > max_n:
        n = max_n
    axis = np.linspace(0.0, domain_mm, n)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    k = 2.0 * np.pi / max(L_mm, 1e-6)
    F = evaluate_tpms(lattice_type, k, X, Y, Z)
    solid_mask = np.abs(F) <= tau
    sf_eff = float(np.mean(solid_mask))
    void_mask = ~solid_mask

    mis = compute_max_inscribed_sphere_pore_size(
        void_mask=void_mask,
        resolution_mm=float(axis[1] - axis[0]),
        boundary_guard_mm=cfg.boundary_guard_cells * L_mm,
        require_sphere_within_domain=True,
    )
    pore_mis = mis.max_diameter_mm
    if pore_mis <= 0.0:
        # Fallback without guard in extreme small domains.
        mis = compute_max_inscribed_sphere_pore_size(
            void_mask=void_mask,
            resolution_mm=float(axis[1] - axis[0]),
            boundary_guard_mm=0.0,
            require_sphere_within_domain=True,
        )
        pore_mis = mis.max_diameter_mm
    return pore_mis, sf_eff


def calibrate_tpms_point(
    lattice_type: str,
    target_pore_mm: float,
    target_solid_fraction: float,
    *,
    initial_L_mm: float | None = None,
    initial_tau: float | None = None,
    config: CalibrationConfig | None = None,
    seed_table: dict[str, Any] | None = None,
    use_seed_table: bool = True,
) -> CalibrationPointResult:
    """
    Calibrate TPMS parameters (L, tau) to achieve a target pore size and solid fraction.

    Parameters
    ----------
    lattice_type : str
        The type of TPMS lattice (e.g., 'gyroid', 'diamond').
    target_pore_mm : float
        The desired maximum inscribed sphere pore diameter in mm.
    target_solid_fraction : float
        The desired solid volume fraction (0.0 to 1.0).
    initial_L_mm : float, optional
        Initial guess for the lattice period L in mm, by default None.
    initial_tau : float, optional
        Initial guess for the level-set threshold tau, by default None.
    config : CalibrationConfig, optional
        Configuration for the solver tolerances and limits, by default None.
    seed_table : dict, optional
        A dictionary containing historical calibration seeds for faster convergence, by default None.
    use_seed_table : bool, optional
        Whether to attempt to look up initial guesses in the seed table, by default True.

    Returns
    -------
    CalibrationPointResult
        A dataclass containing the final calibrated parameters and convergence history.

    Raises
    ------
    ValueError
        If target_pore_mm <= 0 or target_solid_fraction is not in (0, 1).
    """
    cfg = config or CalibrationConfig()
    if target_pore_mm <= 0:
        raise ValueError("target_pore_mm must be > 0")
    if not (0.0 < target_solid_fraction < 1.0):
        raise ValueError("target_solid_fraction must be within (0, 1)")

    seed_src = "heuristic"
    L0 = initial_L_mm
    t0 = initial_tau
    if use_seed_table:
        table = seed_table if seed_table is not None else load_calibration_seed_table()
        L_seed, t_seed = _nearest_seed_guess(
            lattice_type=lattice_type,
            target_pore_mm=target_pore_mm,
            target_solid_fraction=target_solid_fraction,
            seed_table=table,
        )
        if L0 is None and L_seed is not None:
            L0 = L_seed
            seed_src = "seed_table"
        if t0 is None and t_seed is not None:
            t0 = t_seed
            seed_src = "seed_table"
    if L0 is None or t0 is None:
        hL, ht = _heuristic_seed(target_pore_mm, target_solid_fraction)
        L0 = hL if L0 is None else L0
        t0 = ht if t0 is None else t0
        if seed_src != "seed_table":
            seed_src = "heuristic"

    L = float(np.clip(L0, cfg.min_L_mm, cfg.max_L_mm))
    tau = float(np.clip(t0, cfg.min_tau, cfg.max_tau))
    history: list[CalibrationIteration] = []
    notes: list[str] = []
    converged = False
    hit_bounds = False

    for it in range(1, cfg.max_iterations + 1):
        pore_mis, sf_eff = _sample_tpms_metrics(lattice_type, L, tau, cfg)
        r_pore = pore_mis - target_pore_mm
        r_sf = sf_eff - target_solid_fraction
        history.append(
            CalibrationIteration(
                iteration=it,
                L_mm=L,
                tau=tau,
                pore_mis_mm=pore_mis,
                solid_fraction_effective=sf_eff,
                residual_pore_mm=r_pore,
                residual_solid_fraction=r_sf,
            )
        )
        if abs(r_pore) <= cfg.pore_tolerance_mm and abs(r_sf) <= cfg.solid_fraction_tolerance:
            converged = True
            break

        dL = max(cfg.fd_relative_step_L * L, 1e-4)
        dt = max(cfg.fd_absolute_step_tau, 1e-4)
        pore_L, sf_L = _sample_tpms_metrics(lattice_type, L + dL, tau, cfg)
        pore_t, sf_t = _sample_tpms_metrics(lattice_type, L, tau + dt, cfg)

        J = np.array(
            [
                [(pore_L - pore_mis) / dL, (pore_t - pore_mis) / dt],
                [(sf_L - sf_eff) / dL, (sf_t - sf_eff) / dt],
            ],
            dtype=float,
        )
        r = np.array([r_pore, r_sf], dtype=float)

        use_fallback = (not np.all(np.isfinite(J))) or abs(np.linalg.det(J)) < 1e-10
        if use_fallback:
            # Robust fallback: target pore by scaling L, target SF via tau correction.
            L_new = L * (target_pore_mm / max(pore_mis, 1e-6))
            tau_new = tau - 0.6 * r_sf
        else:
            step = np.linalg.solve(J, r)
            L_new = L - cfg.damping * step[0]
            tau_new = tau - cfg.damping * step[1]

        L_clamped = float(np.clip(L_new, cfg.min_L_mm, cfg.max_L_mm))
        tau_clamped = float(np.clip(tau_new, cfg.min_tau, cfg.max_tau))
        hit_bounds = hit_bounds or (L_clamped != L_new) or (tau_clamped != tau_new)
        L, tau = L_clamped, tau_clamped

        if it >= 3:
            prev = history[-2]
            if abs(prev.residual_pore_mm - r_pore) < 1e-5 and abs(prev.residual_solid_fraction - r_sf) < 1e-5:
                notes.append("Calibration stagnated before tolerances were met.")
                break

    final = history[-1]
    if not converged and not notes:
        notes.append("Reached max_iterations without full convergence.")

    return CalibrationPointResult(
        lattice_type=lattice_type,
        target_pore_mm=target_pore_mm,
        target_solid_fraction=target_solid_fraction,
        calibrated_L_mm=final.L_mm,
        calibrated_tau=final.tau,
        converged=converged,
        iterations=len(history),
        residual_pore_mm=final.residual_pore_mm,
        residual_solid_fraction=final.residual_solid_fraction,
        measured_pore_mm=final.pore_mis_mm,
        measured_solid_fraction=final.solid_fraction_effective,
        hit_bounds=hit_bounds,
        history=history,
        seed_source=seed_src,
        notes=notes,
    )


def calibrate_period_for_pore_at_wall(
    lattice_type: str,
    target_pore_mm: float,
    wall_thickness_mm: float,
    *,
    initial_L_mm: float | None = None,
    config: CalibrationConfig | None = None,
) -> CalibrationPointResult:
    """
    Point-wise unit-cell calibration with **fixed wall thickness**.

    Holds the physical wall recipe ``tau = pi * w / L`` (same as field-driven
    ``density_mode=wall_thickness``) and solves only for period ``L`` so the
    uniform-cell MIS pore matches ``target_pore_mm``.

    This is the intended control-point recipe for graded parts: calibrate each
    knot as a uniform cell, then interpolate L(r) / w(r) without gradient
    re-calibration.
    """
    cfg = config or CalibrationConfig()
    if target_pore_mm <= 0:
        raise ValueError("target_pore_mm must be > 0")
    if wall_thickness_mm <= 0:
        raise ValueError("wall_thickness_mm must be > 0")

    L = float(
        np.clip(
            initial_L_mm if initial_L_mm is not None else 1.8 * target_pore_mm,
            cfg.min_L_mm,
            cfg.max_L_mm,
        )
    )
    history: list[CalibrationIteration] = []
    notes: list[str] = [
        f"Fixed wall_thickness_mm={wall_thickness_mm:g}; solve L only (unit-cell)."
    ]
    converged = False
    hit_bounds = False
    tau = float(tau_from_wall_thickness_mm(wall_thickness_mm, L).ravel()[0])

    for it in range(1, cfg.max_iterations + 1):
        tau = float(tau_from_wall_thickness_mm(wall_thickness_mm, L).ravel()[0])
        tau = float(np.clip(tau, cfg.min_tau, cfg.max_tau))
        pore_mis, sf_eff = _sample_tpms_metrics(lattice_type, L, tau, cfg)
        r_pore = pore_mis - target_pore_mm
        history.append(
            CalibrationIteration(
                iteration=it,
                L_mm=L,
                tau=tau,
                pore_mis_mm=pore_mis,
                solid_fraction_effective=sf_eff,
                residual_pore_mm=r_pore,
                residual_solid_fraction=0.0,
            )
        )
        if abs(r_pore) <= cfg.pore_tolerance_mm:
            converged = True
            break

        # Pore scales roughly with L at fixed w/L (approximately); scale L.
        if pore_mis <= 1e-9:
            L_new = L * 1.25
        else:
            L_new = L * (target_pore_mm / pore_mis)
        # Mild damping + finite-difference refine
        dL = max(cfg.fd_relative_step_L * L, 1e-4)
        pore_plus, _ = _sample_tpms_metrics(
            lattice_type,
            L + dL,
            float(tau_from_wall_thickness_mm(wall_thickness_mm, L + dL).ravel()[0]),
            cfg,
        )
        dpdL = (pore_plus - pore_mis) / dL
        if np.isfinite(dpdL) and abs(dpdL) > 1e-8:
            L_newton = L - cfg.damping * r_pore / dpdL
            L_new = 0.5 * L_new + 0.5 * L_newton

        L_clamped = float(np.clip(L_new, cfg.min_L_mm, cfg.max_L_mm))
        hit_bounds = hit_bounds or (L_clamped != L_new)
        L = L_clamped

        if it >= 3:
            prev = history[-2]
            if abs(prev.residual_pore_mm - r_pore) < 1e-5:
                notes.append("Period solve stagnated before pore tolerance.")
                break

    final = history[-1]
    if not converged and "Reached max_iterations" not in " ".join(notes):
        notes.append("Reached max_iterations without pore convergence.")

    return CalibrationPointResult(
        lattice_type=lattice_type,
        target_pore_mm=float(target_pore_mm),
        target_solid_fraction=float(final.solid_fraction_effective),
        calibrated_L_mm=float(final.L_mm),
        calibrated_tau=float(final.tau),
        converged=converged,
        iterations=len(history),
        residual_pore_mm=float(final.residual_pore_mm),
        residual_solid_fraction=0.0,
        measured_pore_mm=float(final.pore_mis_mm),
        measured_solid_fraction=float(final.solid_fraction_effective),
        hit_bounds=hit_bounds,
        history=history,
        seed_source="wall_fixed_period_solve",
        notes=notes,
    )


def calibrate_period_for_pore_at_sf(
    lattice_type: str,
    target_pore_mm: float,
    target_solid_fraction: float,
    *,
    initial_L_mm: float | None = None,
    config: CalibrationConfig | None = None,
) -> CalibrationPointResult:
    """
    Point-wise unit-cell calibration with **fixed solid fraction**.

    Sets ``tau`` from the |F| quantile at ``target_solid_fraction`` (L-invariant
    in phase space) and solves only for period ``L`` so MIS pore matches
    ``target_pore_mm``. Preferred over joint Newton when density is specified
    as solid fraction.
    """
    cfg = config or CalibrationConfig()
    if target_pore_mm <= 0:
        raise ValueError("target_pore_mm must be > 0")
    if not (0.0 < target_solid_fraction < 1.0):
        raise ValueError("target_solid_fraction must be within (0, 1)")

    abs_vals = sample_abs_tpms_phase_distribution(lattice_type)
    tau = float(tau_from_solid_fraction_quantile(abs_vals, target_solid_fraction))
    tau = float(np.clip(tau, cfg.min_tau, cfg.max_tau))

    L = float(
        np.clip(
            initial_L_mm if initial_L_mm is not None else 1.8 * target_pore_mm,
            cfg.min_L_mm,
            cfg.max_L_mm,
        )
    )
    history: list[CalibrationIteration] = []
    notes: list[str] = [
        f"Fixed SF={target_solid_fraction:g} via |F| quantile tau={tau:.4f}; solve L only."
    ]
    converged = False
    hit_bounds = False

    for it in range(1, cfg.max_iterations + 1):
        pore_mis, sf_eff = _sample_tpms_metrics(lattice_type, L, tau, cfg)
        r_pore = pore_mis - target_pore_mm
        r_sf = sf_eff - target_solid_fraction
        history.append(
            CalibrationIteration(
                iteration=it,
                L_mm=L,
                tau=tau,
                pore_mis_mm=pore_mis,
                solid_fraction_effective=sf_eff,
                residual_pore_mm=r_pore,
                residual_solid_fraction=r_sf,
            )
        )
        if abs(r_pore) <= cfg.pore_tolerance_mm and abs(r_sf) <= cfg.solid_fraction_tolerance:
            converged = True
            break

        if pore_mis <= 1e-9:
            L_new = L * 1.25
        else:
            L_new = L * (target_pore_mm / max(pore_mis, 1e-6))

        dL = max(cfg.fd_relative_step_L * L, 1e-4)
        pore_plus, _ = _sample_tpms_metrics(lattice_type, L + dL, tau, cfg)
        dpdL = (pore_plus - pore_mis) / dL
        if np.isfinite(dpdL) and abs(dpdL) > 1e-8:
            L_newton = L - cfg.damping * r_pore / dpdL
            L_new = 0.5 * L_new + 0.5 * L_newton

        L_clamped = float(np.clip(L_new, cfg.min_L_mm, cfg.max_L_mm))
        hit_bounds = hit_bounds or (L_clamped != L_new)
        L = L_clamped

        if it >= 4:
            prev = history[-2]
            if abs(prev.residual_pore_mm - r_pore) < 1e-5:
                notes.append("Period solve stagnated before pore tolerance.")
                break

    final = history[-1]
    if not converged and final.pore_mis_mm > 1e-9:
        # One-shot scale correction (pore ~ linear in L at fixed tau).
        L_fix = float(
            np.clip(
                final.L_mm * (target_pore_mm / final.pore_mis_mm),
                cfg.min_L_mm,
                cfg.max_L_mm,
            )
        )
        pore_fix, sf_fix = _sample_tpms_metrics(lattice_type, L_fix, tau, cfg)
        history.append(
            CalibrationIteration(
                iteration=len(history) + 1,
                L_mm=L_fix,
                tau=tau,
                pore_mis_mm=pore_fix,
                solid_fraction_effective=sf_fix,
                residual_pore_mm=pore_fix - target_pore_mm,
                residual_solid_fraction=sf_fix - target_solid_fraction,
            )
        )
        final = history[-1]
        notes.append("Applied one-shot L scale correction after primary loop.")
        if abs(final.residual_pore_mm) <= cfg.pore_tolerance_mm and abs(
            final.residual_solid_fraction
        ) <= cfg.solid_fraction_tolerance:
            converged = True

    if not converged:
        notes.append("Reached max_iterations without full pore/SF convergence.")

    return CalibrationPointResult(
        lattice_type=lattice_type,
        target_pore_mm=float(target_pore_mm),
        target_solid_fraction=float(target_solid_fraction),
        calibrated_L_mm=float(final.L_mm),
        calibrated_tau=float(tau),
        converged=converged,
        iterations=len(history),
        residual_pore_mm=float(final.residual_pore_mm),
        residual_solid_fraction=float(final.residual_solid_fraction),
        measured_pore_mm=float(final.pore_mis_mm),
        measured_solid_fraction=float(final.solid_fraction_effective),
        hit_bounds=hit_bounds,
        history=history,
        seed_source="sf_fixed_period_solve",
        notes=notes,
    )


def _build_default_anchor_indices(n: int) -> list[int]:
    if n <= 2:
        return list(range(n))
    mid = n // 2
    return sorted(set([0, mid, n - 1]))


def calibrate_tpms_gradient_profile(
    lattice_type: str,
    z_points_mm: list[float] | np.ndarray,
    target_pores_mm: list[float] | np.ndarray,
    target_solid_fractions: list[float] | np.ndarray,
    *,
    mode: str = "hybrid",
    anchor_indices: list[int] | None = None,
    correction_indices: list[int] | None = None,
    config: CalibrationConfig | None = None,
    seed_table: dict[str, Any] | None = None,
    use_seed_table: bool = True,
) -> CalibrationGradientResult:
    """
    Calibrate TPMS parameters over a varying 1D spatial profile (Z-axis).

    Supports 'hybrid' mode (calibrates at anchors and interpolates, then applies 
    corrections) or 'full_profile' (calibrates every point sequentially).

    Parameters
    ----------
    lattice_type : str
        The type of TPMS lattice (e.g., 'gyroid', 'diamond').
    z_points_mm : list of float or ndarray
        Sorted Z-coordinates for the profile points.
    target_pores_mm : list of float or ndarray
        Target pore diameters corresponding to each Z point.
    target_solid_fractions : list of float or ndarray
        Target solid fractions corresponding to each Z point.
    mode : str, optional
        'hybrid' or 'full_profile', by default "hybrid".
    anchor_indices : list of int, optional
        Indices of points to use as initial interpolation anchors in hybrid mode.
    correction_indices : list of int, optional
        Indices of points to apply correction calibration to in hybrid mode.
    config : CalibrationConfig, optional
        Solver configuration.
    seed_table : dict, optional
        Historical seed table for initial guesses.
    use_seed_table : bool, optional
        Whether to use the seed table for anchors, by default True.

    Returns
    -------
    CalibrationGradientResult
        A dataclass containing the profile arrays for calibrated L and tau.

    Raises
    ------
    ValueError
        If input arrays are empty, have mismatched lengths, or if z_points_mm is not sorted.
    """
    cfg = config or CalibrationConfig()
    z = np.asarray(z_points_mm, dtype=float).ravel()
    pores = np.asarray(target_pores_mm, dtype=float).ravel()
    sfs = np.asarray(target_solid_fractions, dtype=float).ravel()
    if z.size == 0:
        raise ValueError("z_points_mm must not be empty")
    if not (z.size == pores.size == sfs.size):
        raise ValueError("z_points_mm, target_pores_mm, target_solid_fractions must match length")
    if np.any(np.diff(z) < 0):
        raise ValueError("z_points_mm must be sorted ascending")

    point_results: list[CalibrationPointResult] = []
    mode_norm = mode.lower().strip()
    if mode_norm not in {"hybrid", "full_profile"}:
        raise ValueError("mode must be 'hybrid' or 'full_profile'")

    if mode_norm == "full_profile":
        Ls: list[float] = []
        taus: list[float] = []
        last_L = None
        last_tau = None
        for p, sf in zip(pores, sfs):
            res = calibrate_tpms_point(
                lattice_type=lattice_type,
                target_pore_mm=float(p),
                target_solid_fraction=float(sf),
                initial_L_mm=last_L,
                initial_tau=last_tau,
                config=cfg,
                seed_table=seed_table,
                use_seed_table=use_seed_table,
            )
            point_results.append(res)
            Ls.append(res.calibrated_L_mm)
            taus.append(res.calibrated_tau)
            last_L, last_tau = res.calibrated_L_mm, res.calibrated_tau
        converged_all = all(r.converged for r in point_results)
        return CalibrationGradientResult(
            mode="full_profile",
            z_points_mm=[float(v) for v in z],
            target_pores_mm=[float(v) for v in pores],
            target_solid_fractions=[float(v) for v in sfs],
            calibrated_L_mm=Ls,
            calibrated_tau=[float(v) for v in taus],
            control_point_results=point_results,
            converged_all=converged_all,
            notes=[],
        )

    anchors = anchor_indices if anchor_indices is not None else _build_default_anchor_indices(int(z.size))
    anchors = sorted(set(int(i) for i in anchors if 0 <= int(i) < z.size))
    if not anchors:
        anchors = [0, int(z.size) - 1]

    anchor_results: dict[int, CalibrationPointResult] = {}
    for idx in anchors:
        r = calibrate_tpms_point(
            lattice_type=lattice_type,
            target_pore_mm=float(pores[idx]),
            target_solid_fraction=float(sfs[idx]),
            config=cfg,
            seed_table=seed_table,
            use_seed_table=use_seed_table,
        )
        anchor_results[idx] = r
        point_results.append(r)

    z_anchor = np.array([z[i] for i in anchors], dtype=float)
    L_anchor = np.array([anchor_results[i].calibrated_L_mm for i in anchors], dtype=float)
    t_anchor = np.array([anchor_results[i].calibrated_tau for i in anchors], dtype=float)

    L_pred = np.interp(z, z_anchor, L_anchor)
    t_pred = np.interp(z, z_anchor, t_anchor)

    if correction_indices is None:
        correction_indices = [i for i in range(z.size) if i not in anchors]
        if len(correction_indices) > 3:
            take = np.linspace(0, len(correction_indices) - 1, 3).astype(int)
            correction_indices = [correction_indices[i] for i in take]
    correction_indices = sorted(set(int(i) for i in correction_indices if 0 <= int(i) < z.size and i not in anchors))

    corrected: dict[int, CalibrationPointResult] = {}
    for idx in correction_indices:
        rr = calibrate_tpms_point(
            lattice_type=lattice_type,
            target_pore_mm=float(pores[idx]),
            target_solid_fraction=float(sfs[idx]),
            initial_L_mm=float(L_pred[idx]),
            initial_tau=float(t_pred[idx]),
            config=cfg,
            seed_table=seed_table,
            use_seed_table=False,
        )
        corrected[idx] = rr
        point_results.append(rr)

    control_idx = sorted(set(anchors + list(corrected.keys())))
    z_ctrl = np.array([z[i] for i in control_idx], dtype=float)
    L_ctrl = np.array(
        [
            corrected[i].calibrated_L_mm if i in corrected else anchor_results[i].calibrated_L_mm
            for i in control_idx
        ],
        dtype=float,
    )
    t_ctrl = np.array(
        [
            corrected[i].calibrated_tau if i in corrected else anchor_results[i].calibrated_tau
            for i in control_idx
        ],
        dtype=float,
    )

    L_final = np.interp(z, z_ctrl, L_ctrl)
    t_final = np.interp(z, z_ctrl, t_ctrl)
    converged_all = all(r.converged for r in point_results)
    notes = [
        "Hybrid mode: anchors calibrated, profile predicted, sparse control-point corrections applied."
    ]
    return CalibrationGradientResult(
        mode="hybrid",
        z_points_mm=[float(v) for v in z],
        target_pores_mm=[float(v) for v in pores],
        target_solid_fractions=[float(v) for v in sfs],
        calibrated_L_mm=[float(v) for v in L_final],
        calibrated_tau=[float(v) for v in t_final],
        control_point_results=point_results,
        converged_all=converged_all,
        notes=notes,
    )


def calibration_result_to_dict(result: CalibrationPointResult | CalibrationGradientResult) -> dict[str, Any]:
    return asdict(result)

