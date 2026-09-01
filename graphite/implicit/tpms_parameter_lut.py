# -*- coding: utf-8 -*-
"""
TPMS parameter lookup table: target (pore MIS, solid fraction) → (L, τ).

Builds on ``calibrate_tpms_point`` so each scaffold type gets its own
calibrated period / threshold instead of the lattice-agnostic pore→L heuristic.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from graphite.implicit.calibration import (
    CalibrationConfig,
    CalibrationPointResult,
    calibrate_tpms_point,
    load_calibration_seed_table,
    update_seed_table_with_result,
)

DEFAULT_LUT_PATH = Path(__file__).with_name("tpms_parameter_lut.json")

STREAMLIT_TPMS_TYPES = (
    "gyroid",
    "diamond",
    "schwarz-diamond",
    "schwarz-p",
    "neovius",
    "split-p",
)


@dataclass(frozen=True)
class TpmsParameterLookup:
    lattice_type: str
    target_pore_mm: float
    target_solid_fraction: float
    L_mm: float
    tau: float
    measured_pore_mm: float
    measured_solid_fraction: float
    converged: bool
    residual_pore_mm: float
    residual_solid_fraction: float
    notes: list[str]


def adaptive_calibration_config(target_pore_mm: float) -> CalibrationConfig:
    """Scale sampling so grids stay under ~80³ even if L grows mid-solve."""
    pore = max(float(target_pore_mm), 0.05)
    max_L = max(90.0, 4.5 * pore)
    max_domain = 1.5 * max_L
    res = float(np.clip(max_domain / 72.0, 0.05, 2.5))
    return CalibrationConfig(
        pore_tolerance_mm=max(0.05, 0.03 * pore),
        solid_fraction_tolerance=0.03,
        max_iterations=18,
        damping=0.55,
        sample_resolution_mm=res,
        sample_cells=1.5,
        boundary_guard_cells=0.35,
        min_voxels_per_period=20,
        min_L_mm=0.05,
        max_L_mm=max_L,
        min_tau=0.02,
        max_tau=3.5,
    )


def _sf_quantile_tau(lattice_type: str, target_solid_fraction: float) -> float:
    from graphite.implicit.calibration import (
        sample_abs_tpms_phase_distribution,
        tau_from_solid_fraction_quantile,
    )

    abs_vals = sample_abs_tpms_phase_distribution(lattice_type)
    return float(tau_from_solid_fraction_quantile(abs_vals, target_solid_fraction))


def load_tpms_parameter_lut(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_LUT_PATH
    if not p.exists():
        return {"version": 1, "entries": [], "meta": {}}
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {"version": 1, "entries": [], "meta": {}}
    data.setdefault("version", 1)
    data.setdefault("entries", [])
    data.setdefault("meta", {})
    return data


def save_tpms_parameter_lut(
    lut: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    p = Path(path) if path else DEFAULT_LUT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(lut, fh, indent=2)
    return p


def calibrate_lut_point(
    lattice_type: str,
    target_pore_mm: float,
    target_solid_fraction: float,
    *,
    use_seed_table: bool = True,
    persist_seed: bool = True,
) -> TpmsParameterLookup:
    cfg = adaptive_calibration_config(target_pore_mm)
    tau0 = _sf_quantile_tau(lattice_type, target_solid_fraction)
    L0 = max(float(target_pore_mm) * 1.6, cfg.min_L_mm)
    if use_seed_table:
        from graphite.implicit.calibration import _nearest_seed_guess, load_calibration_seed_table

        L_seed, t_seed = _nearest_seed_guess(
            lattice_type=lattice_type,
            target_pore_mm=target_pore_mm,
            target_solid_fraction=target_solid_fraction,
            seed_table=load_calibration_seed_table(),
        )
        if L_seed is not None and abs(float(L_seed) - target_pore_mm) < 2.5 * target_pore_mm:
            L0 = float(L_seed)
        if t_seed is not None and 0.02 <= float(t_seed) <= 3.0:
            tau0 = float(t_seed)

    try:
        result: CalibrationPointResult = calibrate_tpms_point(
            lattice_type=lattice_type,
            target_pore_mm=target_pore_mm,
            target_solid_fraction=target_solid_fraction,
            initial_L_mm=L0,
            initial_tau=tau0,
            config=cfg,
            use_seed_table=False,
        )
    except MemoryError:
        return TpmsParameterLookup(
            lattice_type=lattice_type.lower(),
            target_pore_mm=float(target_pore_mm),
            target_solid_fraction=float(target_solid_fraction),
            L_mm=float(L0),
            tau=float(tau0),
            measured_pore_mm=0.0,
            measured_solid_fraction=0.0,
            converged=False,
            residual_pore_mm=float("nan"),
            residual_solid_fraction=float("nan"),
            notes=["MemoryError during metric sampling"],
        )
    if persist_seed and result.converged:
        update_seed_table_with_result(result, persist=True)
    return TpmsParameterLookup(
        lattice_type=str(result.lattice_type).lower(),
        target_pore_mm=float(result.target_pore_mm),
        target_solid_fraction=float(result.target_solid_fraction),
        L_mm=float(result.calibrated_L_mm),
        tau=float(result.calibrated_tau),
        measured_pore_mm=float(result.measured_pore_mm),
        measured_solid_fraction=float(result.measured_solid_fraction),
        converged=bool(result.converged),
        residual_pore_mm=float(result.residual_pore_mm),
        residual_solid_fraction=float(result.residual_solid_fraction),
        notes=list(result.notes),
    )


def upsert_lut_entry(lut: dict[str, Any], entry: TpmsParameterLookup) -> dict[str, Any]:
    entries: list[dict[str, Any]] = lut.setdefault("entries", [])
    key = (
        entry.lattice_type.lower(),
        round(entry.target_pore_mm, 6),
        round(entry.target_solid_fraction, 6),
    )
    payload = asdict(entry)
    for i, old in enumerate(entries):
        old_key = (
            str(old.get("lattice_type", "")).lower(),
            round(float(old.get("target_pore_mm", 0.0)), 6),
            round(float(old.get("target_solid_fraction", 0.0)), 6),
        )
        if old_key == key:
            entries[i] = payload
            return lut
    entries.append(payload)
    return lut


def _entries_for_lattice(lut: dict[str, Any], lattice_type: str) -> list[dict[str, Any]]:
    want = lattice_type.lower()
    return [
        e
        for e in lut.get("entries", [])
        if str(e.get("lattice_type", "")).lower() == want and bool(e.get("converged", False))
    ]


def lookup_tpms_parameters(
    lattice_type: str,
    target_pore_mm: float,
    target_solid_fraction: float,
    *,
    lut: dict[str, Any] | None = None,
    lut_path: str | Path | None = None,
) -> dict[str, float | str]:
    """
    Interpolate (L, τ) for a lattice at (pore, SF).

    Uses bilinear interpolation on a regular (pore × SF) grid when available;
    otherwise nearest-neighbor among converged entries.
    """
    table = lut if lut is not None else load_tpms_parameter_lut(lut_path)
    entries = _entries_for_lattice(table, lattice_type)
    if not entries:
        raise KeyError(f"No converged LUT entries for lattice_type={lattice_type!r}")

    pores = sorted({float(e["target_pore_mm"]) for e in entries})
    sfs = sorted({float(e["target_solid_fraction"]) for e in entries})
    grid: dict[tuple[float, float], dict[str, Any]] = {
        (float(e["target_pore_mm"]), float(e["target_solid_fraction"])): e for e in entries
    }

    # Prefer bilinear when the query sits inside a full rectangular grid.
    if len(pores) >= 2 and len(sfs) >= 2:
        complete = all((p, s) in grid for p in pores for s in sfs)
        if complete:
            L_grid = np.array([[grid[(p, s)]["L_mm"] for s in sfs] for p in pores], dtype=np.float64)
            tau_grid = np.array([[grid[(p, s)]["tau"] for s in sfs] for p in pores], dtype=np.float64)
            pore_c = float(np.clip(target_pore_mm, pores[0], pores[-1]))
            sf_c = float(np.clip(target_solid_fraction, sfs[0], sfs[-1]))
            # Manual bilinear to avoid hard scipy dependency in this path.
            ip = int(np.searchsorted(pores, pore_c, side="right") - 1)
            iq = int(np.searchsorted(sfs, sf_c, side="right") - 1)
            ip = int(np.clip(ip, 0, len(pores) - 2))
            iq = int(np.clip(iq, 0, len(sfs) - 2))
            p0, p1 = pores[ip], pores[ip + 1]
            s0, s1 = sfs[iq], sfs[iq + 1]
            tp = 0.0 if p1 <= p0 else (pore_c - p0) / (p1 - p0)
            ts = 0.0 if s1 <= s0 else (sf_c - s0) / (s1 - s0)

            def _bilerp(a00, a01, a10, a11):
                return (
                    a00 * (1 - tp) * (1 - ts)
                    + a01 * (1 - tp) * ts
                    + a10 * tp * (1 - ts)
                    + a11 * tp * ts
                )

            return {
                "lattice_type": lattice_type.lower(),
                "L_mm": float(
                    _bilerp(
                        L_grid[ip, iq],
                        L_grid[ip, iq + 1],
                        L_grid[ip + 1, iq],
                        L_grid[ip + 1, iq + 1],
                    )
                ),
                "tau": float(
                    _bilerp(
                        tau_grid[ip, iq],
                        tau_grid[ip, iq + 1],
                        tau_grid[ip + 1, iq],
                        tau_grid[ip + 1, iq + 1],
                    )
                ),
                "method": "bilinear",
                "target_pore_mm": float(target_pore_mm),
                "target_solid_fraction": float(target_solid_fraction),
            }

    # Nearest neighbor fallback
    best = min(
        entries,
        key=lambda e: abs(float(e["target_pore_mm"]) - target_pore_mm)
        + 2.0 * abs(float(e["target_solid_fraction"]) - target_solid_fraction),
    )
    return {
        "lattice_type": lattice_type.lower(),
        "L_mm": float(best["L_mm"]),
        "tau": float(best["tau"]),
        "method": "nearest",
        "target_pore_mm": float(target_pore_mm),
        "target_solid_fraction": float(target_solid_fraction),
        "matched_pore_mm": float(best["target_pore_mm"]),
        "matched_solid_fraction": float(best["target_solid_fraction"]),
    }


def build_tpms_parameter_lut(
    *,
    lattice_types: tuple[str, ...] = STREAMLIT_TPMS_TYPES,
    pores_mm: tuple[float, ...] = (3.175, 6.35, 12.7, 25.4),
    solid_fractions: tuple[float, ...] = (0.15, 0.25, 0.33),
    lut_path: str | Path | None = None,
    persist_seeds: bool = True,
) -> dict[str, Any]:
    """Calibrate a dense grid and write ``tpms_parameter_lut.json``."""
    path = Path(lut_path) if lut_path else DEFAULT_LUT_PATH
    lut = load_tpms_parameter_lut(path)
    lut["meta"] = {
        "lattice_types": list(lattice_types),
        "pores_mm": list(pores_mm),
        "solid_fractions": list(solid_fractions),
        "controls": ["L_mm", "tau"],
        "targets": ["MIS_pore_mm", "solid_fraction"],
        "seed_table": str(Path(__file__).with_name("calibration_seed_table.json").name),
    }

    total = len(lattice_types) * len(pores_mm) * len(solid_fractions)
    done = 0
    for lattice in lattice_types:
        for pore in pores_mm:
            for sf in solid_fractions:
                done += 1
                print(
                    f"[{done}/{total}] {lattice}  pore={pore:g}  sf={sf:.2f} ...",
                    flush=True,
                )
                try:
                    entry = calibrate_lut_point(
                        lattice,
                        pore,
                        sf,
                        persist_seed=persist_seeds,
                    )
                except Exception as exc:  # noqa: BLE001 — keep grid build alive
                    entry = TpmsParameterLookup(
                        lattice_type=lattice.lower(),
                        target_pore_mm=float(pore),
                        target_solid_fraction=float(sf),
                        L_mm=float("nan"),
                        tau=float("nan"),
                        measured_pore_mm=0.0,
                        measured_solid_fraction=0.0,
                        converged=False,
                        residual_pore_mm=float("nan"),
                        residual_solid_fraction=float("nan"),
                        notes=[f"Exception: {exc}"],
                    )
                upsert_lut_entry(lut, entry)
                save_tpms_parameter_lut(lut, path)
                status = "OK" if entry.converged else "FAIL"
                print(
                    f"  {status}: L={entry.L_mm:.3f} tau={entry.tau:.4f} "
                    f"meas_pore={entry.measured_pore_mm:.3f} meas_sf={entry.measured_solid_fraction:.3f}",
                    flush=True,
                )

    # Keep seed table warm for future solves
    _ = load_calibration_seed_table()
    return lut
