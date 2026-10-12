"""
Graphite Explicit Interlinked — Universal Analytical PAM Dimensioning Engine

Provides exact closed-form 2-of-4 parameter dimensioning across all Polycatenated
Architected Material (PAM) unit cell architectures:
  - a: Unit cell pitch (mm)
  - d: Strut / wire diameter (mm)
  - Delta: Inter-cage surface clearance (mm)
  - Vf: Solid volume fraction (0 < Vf < 1)

Mathematical Scaling Laws:
  1. Delta = kappa * a - d
  2. Vf = C_geom * (d / a)^2
  3. Jamming limit: Vf_max = C_geom * kappa^2 (beyond which Delta <= 0)

Exact Inversion:
  xi = d / a = sqrt(Vf / C_geom)
  a = Delta / (kappa - xi)
  d = a * xi
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from .cell import InterlinkedCell, InterlinkedRegistry


DEFAULT_WARNING_CLEARANCE: float = 0.250  # 250 um (0.25 mm) recommended DfAM clearance


class JammingLimitExceededError(ValueError):
    """Raised when a requested solid fraction exceeds the theoretical topological jamming limit."""
    pass


@dataclass(frozen=True)
class PAMDimensionResult:
    """
    Result payload of universal PAM analytical dimensioning.

    Attributes:
        cell_name: Canonical or registered name of the cell.
        parent_network: Underlying crystallographic network ('pcu', 'bcc', 'dia', etc.).
        pitch: Unit cell translation pitch a (mm).
        wire_diameter: Strut/wire outer diameter d (mm).
        wire_radius: Strut/wire radius r = d / 2 (mm).
        clearance: Predicted minimum inter-cage surface clearance Delta (mm).
        solid_fraction: Solid volume fraction Vf in [0, 1].
        solid_fraction_percent: Solid volume fraction in percent (0 - 100%).
        aspect_ratio: Wire-to-pitch ratio xi = d / a.
        kappa: Linear clearance coefficient (Delta = kappa * a - d).
        c_geom: Volumetric aspect ratio coefficient (Vf = c_geom * (d / a)^2).
        max_solid_fraction: Theoretical jamming density Vf_max = c_geom * kappa^2.
        max_solid_fraction_percent: Jamming density in percent.
        warning_clearance: Active warning clearance threshold (mm).
        is_feasible: True if Delta > 0 and Vf < Vf_max.
        warnings: List of diagnostic or DfAM warning messages.
    """
    cell_name: str
    parent_network: str
    pitch: float
    wire_diameter: float
    wire_radius: float
    clearance: float
    solid_fraction: float
    solid_fraction_percent: float
    aspect_ratio: float
    kappa: float
    c_geom: float
    max_solid_fraction: float
    max_solid_fraction_percent: float
    cage_diameter: float = 0.0
    warning_clearance: float = DEFAULT_WARNING_CLEARANCE
    is_feasible: bool = True
    warnings: list[str] = field(default_factory=list)


class PAMDimensioner:
    """
    Universal analytical dimensioning and inverse DfAM solver for PAM structures.
    """

    @classmethod
    def resolve_cell(cls, cell: InterlinkedCell | str | type[InterlinkedCell]) -> InterlinkedCell:
        """Resolve a cell instance from string tag, class, or existing instance."""
        if isinstance(cell, str):
            cell_cls = InterlinkedRegistry.get(cell)
            return cell_cls() if callable(cell_cls) else cell_cls
        elif isinstance(cell, type):
            return cell()
        elif isinstance(cell, InterlinkedCell):
            return cell
        raise TypeError(f"Cannot resolve cell of type: {type(cell)}")

    @classmethod
    def compute_c_geom(cls, cell: InterlinkedCell | str) -> float:
        """
        Compute or retrieve the volumetric aspect ratio coefficient C_geom:
            Vf = C_geom * (d / a)^2

        By definition, C_geom is the total solid volume within one conventional unit cell
        when unit cell pitch a = 1.0 mm and wire diameter d = 1.0 mm.
        """
        inst = cls.resolve_cell(cell)

        # 1. Check if the cell class declares an analytical cached constant
        declared = getattr(inst, "SOLID_VOLUME_COEFFICIENT", None) or getattr(inst, "solid_volume_coefficient", None)
        if declared is not None and float(declared) > 0.0:
            return float(declared)

        # 2. Dynamic Integration from basis particles in conventional unit cell at a = 1.0
        network = getattr(inst, "parent_network", "pcu").lower()

        # Determine site positions in conventional unit cell of size a = 1.0
        if network == "dia":
            # Diamond cubic: 8 sites per conventional unit cube
            fcc = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
            sites = []
            for f in fcc:
                sites.append((np.array(f, dtype=np.float64), "A"))
                sites.append((np.array(f, dtype=np.float64) + np.array([0.25, 0.25, 0.25]), "B"))
        elif network == "bcc":
            # Body-centered cubic: 2 sites per unit cube
            sites = [
                (np.array([0.0, 0.0, 0.0]), "A"),
                (np.array([0.5, 0.5, 0.5]), "B"),
            ]
        else:
            # Simple cubic or 2D: 1 site
            sites = [(np.array([0.0, 0.0, 0.0]), "A")]

        total_solid_vol = 0.0
        d_unit = 1.0  # Normalized wire diameter
        r_unit = d_unit / 2.0

        for site_orig, sub in sites:
            try:
                parts = inst.instantiate_site(
                    grid_index=(0, 0, 0),
                    site_origin=site_orig,
                    cell_pitch=1.0,
                    context={"sublattice": sub},
                )
            except Exception:
                parts = inst.instantiate_site(
                    grid_index=(0, 0, 0),
                    site_origin=site_orig,
                    cell_pitch=1.0,
                )

            for p in parts:
                geom = p.geometry
                g_type = getattr(geom, "geometry_type", "").lower()
                meta = getattr(geom, "metadata", {})

                if "ring" in g_type or "major_radius" in meta:
                    R = float(meta.get("major_radius", getattr(geom, "bounding_radius", 0.25)))
                    # Torus volume = 2 * pi^2 * R * r^2 = 2 * pi^2 * R * (d/2)^2 = (pi^2 / 2) * R * d^2
                    v_torus = 0.5 * (np.pi ** 2) * R * (d_unit ** 2)
                    total_solid_vol += v_torus
                elif hasattr(geom, "struts") and len(geom.struts) > 0:
                    nodes = geom.nodes
                    struts = geom.struts
                    diffs = nodes[struts[:, 1]] - nodes[struts[:, 0]]
                    lengths = np.linalg.norm(diffs, axis=1)
                    tot_len = float(np.sum(lengths))
                    # Cylinder volume = pi * r^2 * L = (pi / 4) * d^2 * L
                    v_cyls = 0.25 * np.pi * (d_unit ** 2) * tot_len
                    total_solid_vol += v_cyls

        # In conventional cell of size a = 1.0, V_cell = 1.0^3 = 1.0
        return float(total_solid_vol)

    @classmethod
    def get_kappa(cls, cell: InterlinkedCell | str) -> float:
        """Retrieve the linear clearance coefficient kappa."""
        inst = cls.resolve_cell(cell)
        kappa = getattr(inst, "CLEARANCE_KAPPA", None) or getattr(inst, "clearance_kappa", None)
        if kappa is None:
            raise AttributeError(f"Cell '{getattr(inst, 'name', cell)}' does not define CLEARANCE_KAPPA.")
        return float(kappa)

    @classmethod
    def get_jamming_limit(cls, cell: InterlinkedCell | str) -> float:
        """Calculate theoretical maximum solid volume fraction Vf_max = C_geom * kappa^2."""
        kappa = cls.get_kappa(cell)
        c_geom = cls.compute_c_geom(cell)
        return float(c_geom * (kappa ** 2))

    @classmethod
    def compute_cage_diameter(cls, cell_inst: Any, pitch: float) -> float:
        """Compute the physical bounding outer diameter of an instantiated cage in mm."""
        if math.isnan(pitch) or math.isinf(pitch) or pitch <= 0.0:
            return 0.0
        try:
            parts = cell_inst.instantiate_site((0, 0, 0), np.zeros(3, dtype=np.float64), pitch)
            if parts:
                return float(2.0 * parts[0].bounding_radius)
        except Exception:
            pass
        if hasattr(cell_inst, "edge_ratio"):
            return float(pitch * cell_inst.edge_ratio * math.sqrt(6.0) / 2.0)
        return float(pitch)

    @classmethod
    def _normalize_vf(cls, vf: float) -> float:
        """Normalize solid fraction to [0, 1]. Converts percentages > 1.0 (e.g. 12.0 -> 0.12)."""
        val = float(vf)
        if val < 0.0:
            raise ValueError(f"Solid volume fraction cannot be negative: {vf}")
        if val > 1.0:
            # User provided a percentage (e.g. 10.0 for 10%)
            val = val / 100.0
        return val

    @classmethod
    def solve_from_clearance_and_vf(
        cls,
        cell: InterlinkedCell | str,
        clearance: float,
        solid_fraction: float,
        warning_clearance: float = DEFAULT_WARNING_CLEARANCE,
        raise_on_jamming: bool = True,
    ) -> PAMDimensionResult:
        """
        Inverse DfAM solver: Determine (pitch a, wire diameter d) from target clearance and solid fraction.

        Formulas:
            xi = d / a = sqrt(Vf / C_geom)
            a = Delta / (kappa - xi)
            d = a * xi
        """
        inst = cls.resolve_cell(cell)
        name = getattr(inst, "name", str(cell))
        network = getattr(inst, "parent_network", "pcu")
        delta = float(clearance)
        vf = cls._normalize_vf(solid_fraction)
        kappa = cls.get_kappa(inst)
        c_geom = cls.compute_c_geom(inst)
        vf_max = float(c_geom * (kappa ** 2))

        warnings: list[str] = []
        is_feasible = True

        if vf >= vf_max - 1e-9:
            is_feasible = False
            msg = (
                f"Jamming Limit Exceeded: Target Vf ({vf * 100.0:.2f}%) exceeds the theoretical "
                f"topological jamming limit ({vf_max * 100.0:.2f}%) for '{name}'. "
                f"No positive clearance is geometrically possible."
            )
            if raise_on_jamming:
                raise JammingLimitExceededError(msg)
            warnings.append(msg)
            # In non-raising mode, compute limit state
            xi = kappa
            pitch = float("inf")
            d = float("inf")
        else:
            xi = math.sqrt(vf / c_geom)
            denom = kappa - xi
            pitch = delta / denom
            d = pitch * xi

        if delta < warning_clearance:
            warnings.append(
                f"Target clearance ({delta:.3f} mm) is below the recommended manufacturing "
                f"warning threshold ({warning_clearance:.3f} mm)."
            )

        cage_d = cls.compute_cage_diameter(inst, pitch) if is_feasible else 0.0

        return PAMDimensionResult(
            cell_name=name,
            parent_network=network,
            pitch=float(pitch),
            wire_diameter=float(d),
            wire_radius=float(d * 0.5),
            clearance=float(delta),
            solid_fraction=float(vf),
            solid_fraction_percent=float(vf * 100.0),
            aspect_ratio=float(xi) if is_feasible else float(kappa),
            kappa=float(kappa),
            c_geom=float(c_geom),
            max_solid_fraction=float(vf_max),
            max_solid_fraction_percent=float(vf_max * 100.0),
            cage_diameter=float(cage_d),
            warning_clearance=float(warning_clearance),
            is_feasible=is_feasible,
            warnings=warnings,
        )

    @classmethod
    def solve_from_clearance_and_wire(
        cls,
        cell: InterlinkedCell | str,
        clearance: float,
        wire_diameter: float,
        warning_clearance: float = DEFAULT_WARNING_CLEARANCE,
    ) -> PAMDimensionResult:
        """
        Tooling / beam limit solver: Determine (pitch a, solid fraction Vf) from clearance and fixed wire diameter.

        Formulas:
            a = (Delta + d) / kappa
            Vf = C_geom * (d / a)^2
        """
        inst = cls.resolve_cell(cell)
        name = getattr(inst, "name", str(cell))
        network = getattr(inst, "parent_network", "pcu")
        delta = float(clearance)
        d = float(wire_diameter)
        kappa = cls.get_kappa(inst)
        c_geom = cls.compute_c_geom(inst)
        vf_max = float(c_geom * (kappa ** 2))

        pitch = (delta + d) / kappa
        xi = d / pitch
        vf = c_geom * (xi ** 2)

        warnings: list[str] = []
        if delta < warning_clearance:
            warnings.append(
                f"Target clearance ({delta:.3f} mm) is below the recommended manufacturing "
                f"warning threshold ({warning_clearance:.3f} mm)."
            )

        cage_d = cls.compute_cage_diameter(inst, pitch)

        return PAMDimensionResult(
            cell_name=name,
            parent_network=network,
            pitch=float(pitch),
            wire_diameter=float(d),
            wire_radius=float(d * 0.5),
            clearance=float(delta),
            solid_fraction=float(vf),
            solid_fraction_percent=float(vf * 100.0),
            aspect_ratio=float(xi),
            kappa=float(kappa),
            c_geom=float(c_geom),
            max_solid_fraction=float(vf_max),
            max_solid_fraction_percent=float(vf_max * 100.0),
            cage_diameter=float(cage_d),
            warning_clearance=float(warning_clearance),
            is_feasible=True,
            warnings=warnings,
        )

    @classmethod
    def solve_from_pitch_and_vf(
        cls,
        cell: InterlinkedCell | str,
        pitch: float,
        solid_fraction: float,
        warning_clearance: float = DEFAULT_WARNING_CLEARANCE,
    ) -> PAMDimensionResult:
        """
        Fixed-envelope solver: Determine (wire diameter d, clearance Delta) from pitch and target solid fraction.

        Formulas:
            xi = sqrt(Vf / C_geom)
            d = a * xi
            Delta = kappa * a - d
        """
        inst = cls.resolve_cell(cell)
        name = getattr(inst, "name", str(cell))
        network = getattr(inst, "parent_network", "pcu")
        a = float(pitch)
        vf = cls._normalize_vf(solid_fraction)
        kappa = cls.get_kappa(inst)
        c_geom = cls.compute_c_geom(inst)
        vf_max = float(c_geom * (kappa ** 2))

        warnings: list[str] = []
        is_feasible = True

        if vf >= vf_max:
            is_feasible = False
            warnings.append(
                f"Requested Vf ({vf * 100.0:.2f}%) exceeds jamming limit ({vf_max * 100.0:.2f}%). "
                f"Clearance is zero or negative (collision)."
            )

        xi = math.sqrt(vf / c_geom)
        d = a * xi
        delta = kappa * a - d

        if delta < 0.0:
            is_feasible = False
        elif delta < warning_clearance:
            warnings.append(
                f"Resulting clearance ({delta:.3f} mm) is below the recommended warning threshold ({warning_clearance:.3f} mm)."
            )

        cage_d = cls.compute_cage_diameter(inst, a) if is_feasible else 0.0

        return PAMDimensionResult(
            cell_name=name,
            parent_network=network,
            pitch=float(a),
            wire_diameter=float(d),
            wire_radius=float(d * 0.5),
            clearance=float(delta),
            solid_fraction=float(vf),
            solid_fraction_percent=float(vf * 100.0),
            aspect_ratio=float(xi),
            kappa=float(kappa),
            c_geom=float(c_geom),
            max_solid_fraction=float(vf_max),
            max_solid_fraction_percent=float(vf_max * 100.0),
            cage_diameter=float(cage_d),
            warning_clearance=float(warning_clearance),
            is_feasible=is_feasible,
            warnings=warnings,
        )

    @classmethod
    def solve_from_pitch_and_wire(
        cls,
        cell: InterlinkedCell | str,
        pitch: float,
        wire_diameter: float,
        warning_clearance: float = DEFAULT_WARNING_CLEARANCE,
    ) -> PAMDimensionResult:
        """
        Forward simulation / validation: Determine (clearance Delta, solid fraction Vf) from pitch and wire diameter.

        Formulas:
            Delta = kappa * a - d
            Vf = C_geom * (d / a)^2
        """
        inst = cls.resolve_cell(cell)
        name = getattr(inst, "name", str(cell))
        network = getattr(inst, "parent_network", "pcu")
        a = float(pitch)
        d = float(wire_diameter)
        kappa = cls.get_kappa(inst)
        c_geom = cls.compute_c_geom(inst)
        vf_max = float(c_geom * (kappa ** 2))

        xi = d / a
        vf = c_geom * (xi ** 2)
        delta = kappa * a - d

        warnings: list[str] = []
        is_feasible = True

        if delta < 0.0:
            is_feasible = False
            warnings.append(f"Components physically collide: clearance Delta = {delta:.4f} mm < 0.")
        elif delta < warning_clearance:
            warnings.append(
                f"Clearance ({delta:.3f} mm) is below the recommended warning threshold ({warning_clearance:.3f} mm)."
            )

        cage_d = cls.compute_cage_diameter(inst, a)

        return PAMDimensionResult(
            cell_name=name,
            parent_network=network,
            pitch=float(a),
            wire_diameter=float(d),
            wire_radius=float(d * 0.5),
            clearance=float(delta),
            solid_fraction=float(vf),
            solid_fraction_percent=float(vf * 100.0),
            aspect_ratio=float(xi),
            kappa=float(kappa),
            c_geom=float(c_geom),
            max_solid_fraction=float(vf_max),
            max_solid_fraction_percent=float(vf_max * 100.0),
            cage_diameter=float(cage_d),
            warning_clearance=float(warning_clearance),
            is_feasible=is_feasible,
            warnings=warnings,
        )

    @classmethod
    def solve(
        cls,
        cell: InterlinkedCell | str,
        *,
        pitch: float | None = None,
        wire_diameter: float | None = None,
        clearance: float | None = None,
        solid_fraction: float | None = None,
        warning_clearance: float = DEFAULT_WARNING_CLEARANCE,
        raise_on_jamming: bool = True,
    ) -> PAMDimensionResult:
        """
        Universal 2-of-4 dispatcher: Provide any valid pair among (pitch, wire_diameter, clearance, solid_fraction).
        """
        inputs = {
            "pitch": pitch,
            "wire_diameter": wire_diameter,
            "clearance": clearance,
            "solid_fraction": solid_fraction,
        }
        specified = [k for k, v in inputs.items() if v is not None]

        if len(specified) != 2:
            raise ValueError(
                f"Exact 2-parameter specification required. Received {len(specified)} parameters: {specified}."
            )

        if "clearance" in specified and "solid_fraction" in specified:
            return cls.solve_from_clearance_and_vf(
                cell,
                clearance=float(clearance),
                solid_fraction=float(solid_fraction),
                warning_clearance=warning_clearance,
                raise_on_jamming=raise_on_jamming,
            )
        elif "clearance" in specified and "wire_diameter" in specified:
            return cls.solve_from_clearance_and_wire(
                cell,
                clearance=float(clearance),
                wire_diameter=float(wire_diameter),
                warning_clearance=warning_clearance,
            )
        elif "pitch" in specified and "solid_fraction" in specified:
            return cls.solve_from_pitch_and_vf(
                cell,
                pitch=float(pitch),
                solid_fraction=float(solid_fraction),
                warning_clearance=warning_clearance,
            )
        elif "pitch" in specified and "wire_diameter" in specified:
            return cls.solve_from_pitch_and_wire(
                cell,
                pitch=float(pitch),
                wire_diameter=float(wire_diameter),
                warning_clearance=warning_clearance,
            )
        elif "pitch" in specified and "clearance" in specified:
            # a and Delta specified -> d = kappa * a - Delta
            inst = cls.resolve_cell(cell)
            kappa = cls.get_kappa(inst)
            d = kappa * float(pitch) - float(clearance)
            return cls.solve_from_pitch_and_wire(
                cell,
                pitch=float(pitch),
                wire_diameter=float(d),
                warning_clearance=warning_clearance,
            )
        elif "wire_diameter" in specified and "solid_fraction" in specified:
            # d and Vf specified -> a = d / sqrt(Vf / C_geom)
            inst = cls.resolve_cell(cell)
            c_geom = cls.compute_c_geom(inst)
            vf = cls._normalize_vf(solid_fraction)
            a = float(wire_diameter) / math.sqrt(vf / c_geom)
            return cls.solve_from_pitch_and_wire(
                cell,
                pitch=float(a),
                wire_diameter=float(wire_diameter),
                warning_clearance=warning_clearance,
            )

        raise ValueError(f"Unsupported parameter pair: {specified}")

    @classmethod
    def compare_architectures(
        cls,
        cells: Sequence[InterlinkedCell | str],
        clearance: float,
        solid_fraction: float,
        warning_clearance: float = DEFAULT_WARNING_CLEARANCE,
    ) -> list[PAMDimensionResult]:
        """
        Evaluate and compare multiple PAM architectures for the same (clearance, solid_fraction) target.
        """
        results = []
        for c in cells:
            try:
                res = cls.solve_from_clearance_and_vf(
                    c,
                    clearance=clearance,
                    solid_fraction=solid_fraction,
                    warning_clearance=warning_clearance,
                    raise_on_jamming=False,
                )
                results.append(res)
            except Exception as e:
                # Capture unresolvable cell
                inst = cls.resolve_cell(c)
                name = getattr(inst, "name", str(c))
                res = PAMDimensionResult(
                    cell_name=name,
                    parent_network=getattr(inst, "parent_network", "unknown"),
                    pitch=float("nan"),
                    wire_diameter=float("nan"),
                    wire_radius=float("nan"),
                    clearance=float(clearance),
                    solid_fraction=cls._normalize_vf(solid_fraction),
                    solid_fraction_percent=cls._normalize_vf(solid_fraction) * 100.0,
                    aspect_ratio=float("nan"),
                    kappa=float("nan"),
                    c_geom=float("nan"),
                    max_solid_fraction=float("nan"),
                    max_solid_fraction_percent=float("nan"),
                    cage_diameter=0.0,
                    warning_clearance=warning_clearance,
                    is_feasible=False,
                    warnings=[str(e)],
                )
                results.append(res)
        return results


def dimension_pam(
    cell: InterlinkedCell | str,
    *,
    pitch: float | None = None,
    wire_diameter: float | None = None,
    clearance: float | None = None,
    solid_fraction: float | None = None,
    warning_clearance: float = DEFAULT_WARNING_CLEARANCE,
    raise_on_jamming: bool = True,
) -> PAMDimensionResult:
    """
    Top-level convenience function for PAM dimensioning.
    Provide any valid pair of parameters to solve the remaining two.
    """
    return PAMDimensioner.solve(
        cell,
        pitch=pitch,
        wire_diameter=wire_diameter,
        clearance=clearance,
        solid_fraction=solid_fraction,
        warning_clearance=warning_clearance,
        raise_on_jamming=raise_on_jamming,
    )
