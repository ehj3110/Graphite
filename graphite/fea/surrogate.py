"""
Offline Parameter Sweep and Constitutive Tensor Surrogate Modeling.

Constructs continuous, positive-definite tensor surrogate models (PCHIP spline,
cubic spline, generalized Gibson-Ashby power-law) over sweeps of relative density
or strut dimensions to provide instantaneous vectorized material evaluation for
macro-scale continuum FEA.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from graphite.fea.homogenization import (
    EngineeringConstants,
    HomogenizationResult,
    RVEGridConfig,
    homogenize_strut_cell,
    homogenize_tpms_cell,
)


@dataclass
class SurrogateCalibrationPoint:
    """A single discrete homogenization result evaluated during the parameter sweep."""

    param_value: float
    solid_fraction: float
    C_tensor: np.ndarray  # shape (6, 6) in Voigt notation
    constants: EngineeringConstants

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_value": float(self.param_value),
            "solid_fraction": float(self.solid_fraction),
            "C_tensor": self.C_tensor.tolist(),
            "constants": asdict(self.constants),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrogateCalibrationPoint:
        return cls(
            param_value=float(data["param_value"]),
            solid_fraction=float(data["solid_fraction"]),
            C_tensor=np.asarray(data["C_tensor"], dtype=np.float64),
            constants=EngineeringConstants(**data["constants"]),
        )


@dataclass
class MaterialTensorSurrogate:
    """
    Continuous surrogate model mapping a scalar geometric parameter to a 6x6 elasticity tensor.

    Attributes
    ----------
    param_name : str
        Name of the driving parameter (e.g. 'solid_fraction', 'strut_radius').
    param_range : tuple[float, float]
        Valid (min, max) range for interpolation.
    sample_points : list[SurrogateCalibrationPoint]
        Discrete calibration points.
    symmetry_type : str
        Lattice crystallographic symmetry: 'cubic', 'orthotropic', or 'anisotropic'.
    fitting_method : str
        Interpolation/fitting method: 'pchip', 'cubic', or 'power_law'.
    """

    param_name: str
    param_range: tuple[float, float]
    sample_points: list[SurrogateCalibrationPoint]
    symmetry_type: Literal["cubic", "orthotropic", "anisotropic"] = "cubic"
    fitting_method: Literal["pchip", "cubic", "power_law"] = "pchip"
    _interpolators: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        """Construct interpolation models for independent components based on symmetry."""
        if len(self.sample_points) < 2:
            raise ValueError(
                f"Surrogate requires at least 2 calibration points, got {len(self.sample_points)}"
            )

        # Sort sample points by param_value
        pts = sorted(self.sample_points, key=lambda p: p.param_value)
        x = np.array([p.param_value for p in pts], dtype=np.float64)

        self._interpolators = {}

        if self.symmetry_type == "cubic":
            # Cubic symmetry has 3 independent stiffnesses: C11, C12, C44
            # Voigt ordering in Graphite: 0:xx, 1:yy, 2:zz, 3:xy, 4:yz, 5:xz
            # C11: mean of diagonal normal terms (C00, C11, C22)
            y_C11 = np.array(
                [np.mean([p.C_tensor[0, 0], p.C_tensor[1, 1], p.C_tensor[2, 2]]) for p in pts]
            )
            # C12: mean of off-diagonal normal couplings (C01, C02, C12)
            y_C12 = np.array(
                [np.mean([p.C_tensor[0, 1], p.C_tensor[0, 2], p.C_tensor[1, 2]]) for p in pts]
            )
            # C44: mean of shear terms (C33, C44, C55)
            y_C44 = np.array(
                [np.mean([p.C_tensor[3, 3], p.C_tensor[4, 4], p.C_tensor[5, 5]]) for p in pts]
            )

            self._interpolators["C11"] = self._create_fitter(x, y_C11)
            self._interpolators["C12"] = self._create_fitter(x, y_C12)
            self._interpolators["C44"] = self._create_fitter(x, y_C44)

        elif self.symmetry_type == "orthotropic":
            # 9 independent components
            indices = [
                ("C00", 0, 0),
                ("C11", 1, 1),
                ("C22", 2, 2),
                ("C01", 0, 1),
                ("C02", 0, 2),
                ("C12", 1, 2),
                ("C33", 3, 3),
                ("C44", 4, 4),
                ("C55", 5, 5),
            ]
            for name, r, c in indices:
                y = np.array([p.C_tensor[r, c] for p in pts])
                self._interpolators[name] = self._create_fitter(x, y)

        elif self.symmetry_type == "anisotropic":
            # All 21 upper-triangular components
            for r in range(6):
                for c in range(r, 6):
                    name = f"C{r}{c}"
                    y = np.array([p.C_tensor[r, c] for p in pts])
                    self._interpolators[name] = self._create_fitter(x, y)

    def _create_fitter(self, x: np.ndarray, y: np.ndarray) -> Any:
        """Create a 1D interpolator or power-law model."""
        method = self.fitting_method.lower()
        if method == "pchip":
            return PchipInterpolator(x, y, extrapolate=True)
        elif method == "cubic":
            return CubicSpline(x, y, bc_type="natural", extrapolate=True)
        elif method == "power_law":
            # Fit ln(y) = ln(alpha) + p * ln(x) for positive data
            pos = (x > 0.0) & (y > 0.0)
            if np.sum(pos) >= 2:
                poly = np.polyfit(np.log(x[pos]), np.log(y[pos]), deg=1)
                p_exp, ln_alpha = poly[0], poly[1]
                alpha = np.exp(ln_alpha)
                return ("power_law", alpha, p_exp)
            else:
                # Fallback to Pchip if data has non-positive entries
                return PchipInterpolator(x, y, extrapolate=True)
        else:
            raise ValueError(f"Unknown fitting_method '{self.fitting_method}'")

    def _eval_fitter(self, fitter: Any, x_arr: np.ndarray) -> np.ndarray:
        if isinstance(fitter, tuple) and fitter[0] == "power_law":
            _, alpha, p_exp = fitter
            return alpha * (np.maximum(x_arr, 1e-8) ** p_exp)
        return fitter(x_arr)

    def evaluate_material_tensor(self, param: float | np.ndarray) -> np.ndarray:
        """
        Evaluate effective constitutive stiffness tensor C^H at given parameter values.

        Parameters
        ----------
        param : float or np.ndarray
            Parameter value(s) within or near param_range.

        Returns
        -------
        np.ndarray
            Shape (6, 6) if param is scalar float.
            Shape (..., 6, 6) matching param.shape if param is an ndarray.
        """
        is_scalar = np.isscalar(param) or (isinstance(param, np.ndarray) and param.ndim == 0)
        x_raw = np.asarray(param, dtype=np.float64)
        orig_shape = x_raw.shape
        x_flat = np.atleast_1d(x_raw).ravel()

        # Clamp slightly to avoid extreme extrapolation blowup
        x_clamped = np.clip(
            x_flat,
            self.param_range[0] * 0.95,
            self.param_range[1] * 1.05,
        )

        n_pts = len(x_clamped)
        C_out = np.zeros((n_pts, 6, 6), dtype=np.float64)

        if self.symmetry_type == "cubic":
            c11 = self._eval_fitter(self._interpolators["C11"], x_clamped)
            c12 = self._eval_fitter(self._interpolators["C12"], x_clamped)
            c44 = self._eval_fitter(self._interpolators["C44"], x_clamped)

            # Ensure positive definiteness: C11 > 0, C44 > 0, C11 - C12 > 0, C11 + 2*C12 > 0
            c11 = np.maximum(c11, 1e-6)
            c44 = np.maximum(c44, 1e-6)
            # Bound c12 so bulk and shear stiffness remain positive
            c12 = np.clip(c12, -0.49 * c11, 0.98 * c11)

            # Diagonal normal terms
            C_out[:, 0, 0] = c11
            C_out[:, 1, 1] = c11
            C_out[:, 2, 2] = c11
            # Normal coupling terms
            C_out[:, 0, 1] = C_out[:, 1, 0] = c12
            C_out[:, 0, 2] = C_out[:, 2, 0] = c12
            C_out[:, 1, 2] = C_out[:, 2, 1] = c12
            # Shear terms (xy, yz, xz)
            C_out[:, 3, 3] = c44
            C_out[:, 4, 4] = c44
            C_out[:, 5, 5] = c44

        elif self.symmetry_type == "orthotropic":
            for name, r, c in [
                ("C00", 0, 0),
                ("C11", 1, 1),
                ("C22", 2, 2),
                ("C01", 0, 1),
                ("C02", 0, 2),
                ("C12", 1, 2),
                ("C33", 3, 3),
                ("C44", 4, 4),
                ("C55", 5, 5),
            ]:
                val = self._eval_fitter(self._interpolators[name], x_clamped)
                if r == c:
                    val = np.maximum(val, 1e-6)
                C_out[:, r, c] = val
                C_out[:, c, r] = val

        elif self.symmetry_type == "anisotropic":
            for r in range(6):
                for c in range(r, 6):
                    name = f"C{r}{c}"
                    val = self._eval_fitter(self._interpolators[name], x_clamped)
                    if r == c:
                        val = np.maximum(val, 1e-6)
                    C_out[:, r, c] = val
                    C_out[:, c, r] = val

        if is_scalar:
            return C_out[0]
        return C_out.reshape(orig_shape + (6, 6))

    def evaluate_engineering_constants(
        self, param: float | np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """
        Evaluate directional engineering elastic constants (E, G, nu, A) at given parameter(s).

        Returns
        -------
        dict[str, np.ndarray | float]
            Dictionary containing 'E_x', 'E_y', 'E_z', 'G_xy', 'G_yz', 'G_zx',
            'nu_xy', 'nu_yx', 'nu_xz', 'nu_zx', 'nu_yz', 'nu_zy',
            'bulk_modulus' (alias 'bulk_K'), 'zener_anisotropy' (alias 'zener_A').
        """
        is_scalar = np.isscalar(param) or (isinstance(param, np.ndarray) and param.ndim == 0)
        C_batch = self.evaluate_material_tensor(param)

        if is_scalar:
            C_batch = C_batch[None, :, :]

        orig_shape = C_batch.shape[:-2]
        C_flat = C_batch.reshape(-1, 6, 6)
        n_pts = C_flat.shape[0]

        # Vectorized batch inversion
        S_flat = np.linalg.inv(C_flat)

        E_x = 1.0 / np.maximum(S_flat[:, 0, 0], 1e-15)
        E_y = 1.0 / np.maximum(S_flat[:, 1, 1], 1e-15)
        E_z = 1.0 / np.maximum(S_flat[:, 2, 2], 1e-15)

        G_xy = 1.0 / np.maximum(S_flat[:, 3, 3], 1e-15)
        G_yz = 1.0 / np.maximum(S_flat[:, 4, 4], 1e-15)
        G_zx = 1.0 / np.maximum(S_flat[:, 5, 5], 1e-15)

        nu_xy = -S_flat[:, 1, 0] / np.maximum(S_flat[:, 0, 0], 1e-15)
        nu_yx = -S_flat[:, 0, 1] / np.maximum(S_flat[:, 1, 1], 1e-15)
        nu_xz = -S_flat[:, 2, 0] / np.maximum(S_flat[:, 0, 0], 1e-15)
        nu_zx = -S_flat[:, 0, 2] / np.maximum(S_flat[:, 2, 2], 1e-15)
        nu_yz = -S_flat[:, 2, 1] / np.maximum(S_flat[:, 1, 1], 1e-15)
        nu_zy = -S_flat[:, 1, 2] / np.maximum(S_flat[:, 2, 2], 1e-15)

        bulk_K = np.sum(C_flat[:, :3, :3], axis=(1, 2)) / 9.0

        denom = C_flat[:, 0, 0] - C_flat[:, 0, 1]
        denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
        zener_A = 2.0 * C_flat[:, 3, 3] / denom

        out = {
            "E_x": E_x.reshape(orig_shape),
            "E_y": E_y.reshape(orig_shape),
            "E_z": E_z.reshape(orig_shape),
            "G_xy": G_xy.reshape(orig_shape),
            "G_yz": G_yz.reshape(orig_shape),
            "G_zx": G_zx.reshape(orig_shape),
            "nu_xy": nu_xy.reshape(orig_shape),
            "nu_yx": nu_yx.reshape(orig_shape),
            "nu_xz": nu_xz.reshape(orig_shape),
            "nu_zx": nu_zx.reshape(orig_shape),
            "nu_yz": nu_yz.reshape(orig_shape),
            "nu_zy": nu_zy.reshape(orig_shape),
            "bulk_modulus": bulk_K.reshape(orig_shape),
            "bulk_K": bulk_K.reshape(orig_shape),
            "zener_anisotropy": zener_A.reshape(orig_shape),
            "zener_A": zener_A.reshape(orig_shape),
        }

        if is_scalar:
            return {k: float(np.asarray(v).ravel()[0]) for k, v in out.items()}
        return out

    def save(self, filepath: str | Path) -> None:
        """Serialize the surrogate model and calibration data to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "param_name": self.param_name,
            "param_range": [float(self.param_range[0]), float(self.param_range[1])],
            "symmetry_type": self.symmetry_type,
            "fitting_method": self.fitting_method,
            "sample_points": [pt.to_dict() for pt in self.sample_points],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filepath: str | Path) -> MaterialTensorSurrogate:
        """Deserialize a surrogate model from a JSON file."""
        p = Path(filepath)
        data = json.loads(p.read_text(encoding="utf-8"))
        pts = [SurrogateCalibrationPoint.from_dict(d) for d in data["sample_points"]]
        return cls(
            param_name=data["param_name"],
            param_range=(float(data["param_range"][0]), float(data["param_range"][1])),
            sample_points=pts,
            symmetry_type=data["symmetry_type"],
            fitting_method=data["fitting_method"],
        )


# ===========================================================================
# Automated Parameter Sweep & Calibration Routines
# ===========================================================================


def build_tpms_homogenization_surrogate(
    lattice_type: str = "Gyroid",
    solid_fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
    is_sheet: bool = True,
    rve_config: RVEGridConfig | None = None,
    fitting_method: Literal["pchip", "cubic", "power_law"] = "pchip",
) -> MaterialTensorSurrogate:
    """
    Run an automated parameter sweep over TPMS solid fractions and construct a tensor surrogate.

    Parameters
    ----------
    lattice_type : str
        TPMS equation type (e.g., 'Gyroid', 'Schwarz-P', 'Schwarz-D', 'Split-P').
    solid_fractions : tuple of float
        Sample points for solid volume fraction sweep.
    is_sheet : bool
        Sheet network if True, skeletal solid if False. By default True.
    rve_config : RVEGridConfig, optional
        Voxel grid configuration.
    fitting_method : str
        'pchip' (monotonic, recommended), 'cubic', or 'power_law'.

    Returns
    -------
    MaterialTensorSurrogate
    """
    if rve_config is None:
        rve_config = RVEGridConfig(resolution=24, solver_backend="cg")

    pts = []
    for sf in sorted(solid_fractions):
        res = homogenize_tpms_cell(
            lattice_type=lattice_type,
            solid_fraction=sf,
            is_sheet=is_sheet,
            config=rve_config,
        )
        pts.append(
            SurrogateCalibrationPoint(
                param_value=float(sf),
                solid_fraction=res.solid_fraction,
                C_tensor=res.C_homogenized,
                constants=res.engineering_constants,
            )
        )

    # Determine symmetry type: standard TPMS families are cubic
    l_type = lattice_type.lower()
    if l_type in ["gyroid", "diamond", "schwarz-diamond", "schwarz-p", "schwarz", "neovius"]:
        sym = "cubic"
    else:
        sym = "orthotropic"

    param_range = (float(min(solid_fractions)), float(max(solid_fractions)))
    return MaterialTensorSurrogate(
        param_name="solid_fraction",
        param_range=param_range,
        sample_points=pts,
        symmetry_type=sym,
        fitting_method=fitting_method,
    )


def build_strut_homogenization_surrogate(
    nodes: np.ndarray,
    struts: np.ndarray,
    radii: tuple[float, ...] = (0.05, 0.08, 0.12, 0.16, 0.20),
    symmetry_type: Literal["cubic", "orthotropic", "anisotropic"] = "cubic",
    rve_config: RVEGridConfig | None = None,
    fitting_method: Literal["pchip", "cubic", "power_law"] = "pchip",
) -> MaterialTensorSurrogate:
    """
    Run an automated parameter sweep over explicit strut radii and construct a tensor surrogate.

    Parameters
    ----------
    nodes : np.ndarray
        Unit cell node coordinates in [0, 1]^3, shape (V, 3).
    struts : np.ndarray
        Strut edge connectivity, shape (S, 2).
    radii : tuple of float
        Sweep values for relative strut radius (r / L).
    symmetry_type : str
        Lattice symmetry class.
    rve_config : RVEGridConfig, optional
        Voxel grid configuration.
    fitting_method : str
        Interpolation method.

    Returns
    -------
    MaterialTensorSurrogate
    """
    if rve_config is None:
        rve_config = RVEGridConfig(resolution=24, solver_backend="cg")

    pts = []
    for r in sorted(radii):
        res = homogenize_strut_cell(
            nodes=nodes,
            struts=struts,
            strut_radius=r,
            config=rve_config,
        )
        pts.append(
            SurrogateCalibrationPoint(
                param_value=float(r),
                solid_fraction=res.solid_fraction,
                C_tensor=res.C_homogenized,
                constants=res.engineering_constants,
            )
        )

    param_range = (float(min(radii)), float(max(radii)))
    return MaterialTensorSurrogate(
        param_name="strut_radius",
        param_range=param_range,
        sample_points=pts,
        symmetry_type=symmetry_type,
        fitting_method=fitting_method,
    )


def build_octet_homogenization_surrogate(
    solid_fractions: tuple[float, ...] = (0.08, 0.12, 0.18, 0.26, 0.36),
    rve_config: RVEGridConfig | None = None,
    fitting_method: Literal["pchip", "cubic", "power_law"] = "pchip",
) -> MaterialTensorSurrogate:
    """
    Run an automated parameter sweep over octet truss solid fractions and construct a tensor surrogate.

    Parameters
    ----------
    solid_fractions : tuple of float, optional
        Target solid fractions, by default (0.08, 0.12, 0.18, 0.26, 0.36).
    rve_config : RVEGridConfig, optional
        RVE grid configuration (resolution, material properties).
    fitting_method : str, optional
        Interpolation method ('pchip', 'cubic', 'power_law'). Default 'pchip'.

    Returns
    -------
    MaterialTensorSurrogate
    """
    from graphite.fea.homogenization import homogenize_octet_cell

    if rve_config is None:
        rve_config = RVEGridConfig(resolution=32, solver_backend="cg")

    pts = []
    for sf in sorted(solid_fractions):
        res = homogenize_octet_cell(
            solid_fraction=sf,
            config=rve_config,
        )
        pts.append(
            SurrogateCalibrationPoint(
                param_value=float(sf),
                solid_fraction=res.solid_fraction,
                C_tensor=res.C_homogenized,
                constants=res.engineering_constants,
            )
        )

    param_range = (float(min(solid_fractions)), float(max(solid_fractions)))
    return MaterialTensorSurrogate(
        param_name="solid_fraction",
        param_range=param_range,
        sample_points=pts,
        symmetry_type="cubic",
        fitting_method=fitting_method,
    )

