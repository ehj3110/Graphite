# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Engine - Spatial Grading & Mechanics.

Provides:
1. Spatial grading field evaluators for 2D lattices (linear along X/Y, radial, attractor curves/points).
2. Gibson-Ashby micromechanics models to invert target stiffness fields E(x, y) into strut widths w(x, y).
3. Constant solid-fraction sizing solvers for variable cell-size grids L(x, y).
"""
from __future__ import annotations

from typing import Callable, Sequence, Union
import numpy as np


def eval_linear_gradient(
    p: Sequence[float] | np.ndarray,
    axis: int = 0,
    p_min: float = 0.0,
    p_max: float = 50.0,
    val_min: float = 1.0,
    val_max: float = 3.0,
) -> float:
    """
    Evaluate a linear gradient along a specific coordinate axis (0=X, 1=Y).
    """
    coord = float(p[axis])
    if p_max <= p_min:
        return float(val_min)
    t = np.clip((coord - p_min) / (p_max - p_min), 0.0, 1.0)
    return float(val_min + t * (val_max - val_min))


def eval_radial_gradient(
    p: Sequence[float] | np.ndarray,
    center: Sequence[float] = (0.0, 0.0),
    r_min: float = 0.0,
    r_max: float = 30.0,
    val_inner: float = 1.0,
    val_outer: float = 3.0,
) -> float:
    """
    Evaluate a radially symmetric gradient from a center point.
    """
    dx = float(p[0]) - float(center[0])
    dy = float(p[1]) - float(center[1])
    r = np.sqrt(dx * dx + dy * dy)
    if r_max <= r_min:
        return float(val_inner)
    t = np.clip((r - r_min) / (r_max - r_min), 0.0, 1.0)
    return float(val_inner + t * (val_outer - val_inner))


def eval_attractor_distance(
    p: Sequence[float] | np.ndarray,
    attractor_pts: np.ndarray,
    lambda_decay: float = 10.0,
    val_near: float = 1.0,
    val_far: float = 3.0,
) -> float:
    """
    Evaluate falloff based on Euclidean distance to nearest attractor point/curve.
    Uses smooth cosine half-window transition over characteristic length lambda_decay.
    """
    pts = np.asarray(attractor_pts, dtype=np.float64)[:, :2]
    p_xy = np.asarray(p, dtype=np.float64)[:2]
    dists = np.linalg.norm(pts - p_xy, axis=1)
    min_dist = float(np.min(dists)) if len(dists) > 0 else 0.0

    L = max(float(lambda_decay), 1e-9)
    t = np.clip(min_dist / L, 0.0, 1.0)
    # Cosine half-window: 1 at d=0, 0 at d>=L
    w = 0.5 * (1.0 + np.cos(np.pi * t))
    return float(val_far + (val_near - val_far) * w)


def stiffness_to_strut_width_bending(
    E_target: float,
    E_solid: float,
    L_strut: float,
    C_bend: float = 1.5,
    w_min: float = 0.6,
    w_max: float = 3.0,
) -> float:
    """
    Gibson-Ashby inversion for bending-dominated cellular solids (auxetic re-entrant, chiral, honeycomb):
        E_target / E_solid = C_bend * (w / L)^3
        => w = L * (E_target / (C_bend * E_solid))^(1/3)
    """
    if E_solid <= 0.0 or L_strut <= 0.0 or C_bend <= 0.0:
        return w_min
    ratio = max(E_target / (C_bend * E_solid), 1e-12)
    w = L_strut * (ratio ** (1.0 / 3.0))
    return float(np.clip(w, w_min, w_max))


def stiffness_to_strut_width_stretching(
    E_target: float,
    E_solid: float,
    L_strut: float,
    C_stretch: float = 1.0,
    w_min: float = 0.6,
    w_max: float = 3.0,
) -> float:
    """
    Inversion for stretch-dominated cellular solids (triangular grid, Kagome):
        E_target / E_solid = C_stretch * (w / L)
        => w = L * (E_target / (C_stretch * E_solid))
    """
    if E_solid <= 0.0 or L_strut <= 0.0 or C_stretch <= 0.0:
        return w_min
    ratio = max(E_target / (C_stretch * E_solid), 1e-12)
    w = L_strut * ratio
    return float(np.clip(w, w_min, w_max))


def width_for_constant_solid_fraction(
    cell_size: float,
    solid_fraction: float = 0.15,
    topology: str = "square",
) -> float:
    """
    Calculate the strut width required to maintain an exact constant solid-fraction
    phi_0 as local cell size L varies.

    Parameters
    ----------
    cell_size : float
        Local unit cell pitch L in mm.
    solid_fraction : float
        Target solid fraction phi_0 in (0, 1).
    topology : str
        'square' / 'grid' or 'triangle' / 'tri' / 'kagome'.
    """
    phi = np.clip(float(solid_fraction), 0.01, 0.85)
    L = float(cell_size)
    topo = topology.lower().strip()

    if "tri" in topo or "kagome" in topo:
        # Equilateral triangle cell area = sqrt(3)/2 * L^2
        # Strut length per cell = 3 * L
        # phi ~ 2 * sqrt(3) * (w / L)
        # Higher-order node overlap correction:
        # phi ~ 2*sqrt(3)*(w/L) - 3*sqrt(3)*(w/L)^2
        # Inverting:
        a_coef = 3.0 * np.sqrt(3.0)
        b_coef = -2.0 * np.sqrt(3.0)
        c_coef = phi
        disc = max(b_coef**2 - 4.0 * a_coef * c_coef, 0.0)
        ratio = (-b_coef - np.sqrt(disc)) / (2.0 * a_coef)
        return float(L * ratio)
    else:
        # Square grid cell area = L^2, strut length per cell = 2*L
        # phi = 2*(w/L) - (w/L)^2
        # => (w/L)^2 - 2*(w/L) + phi = 0
        # => w/L = 1 - sqrt(1 - phi)
        ratio = 1.0 - np.sqrt(max(1.0 - phi, 0.0))
        return float(L * ratio)


# =========================================================================
# 1D Developable Sine Wave Curvature Mapping & Thermoforming Compensation
# =========================================================================

def compute_sine_curvature(
    x: float | Sequence[float] | np.ndarray,
    amplitude: float,
    wavelength: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Compute arc length s(x) and local curvature kappa(x) along a 1D developable sine wave.

    Target surface waveform:
        z(x) = amplitude * sin(2 * pi * x / wavelength)

    First derivative:
        z'(x) = amplitude * (2 * pi / wavelength) * cos(2 * pi * x / wavelength)

    Local curvature:
        kappa(x) = |z''(x)| / [1 + (z'(x))^2]^(3/2)
                 = amplitude * (2*pi/wavelength)^2 * |sin(2*pi*x/wavelength)| / [1 + (z'(x))^2]^(3/2)

    Arc length coordinate:
        s(x) = \\int_0^x sqrt(1 + (z'(u))^2) du

    Parameters
    ----------
    x : float, sequence of float, or ndarray
        Projected coordinate(s) along the sine wave axis in mm.
    amplitude : float
        Wave amplitude A in mm (e.g. 0.2 * wavelength).
    wavelength : float
        Wavelength lambda in mm.

    Returns
    -------
    s : float or ndarray
        Arc length coordinate(s) s(x) in mm from x=0.
    kappa : float or ndarray
        Local curvature kappa(x) in mm^-1.
    """
    import scipy.integrate

    A = float(amplitude)
    lam = float(wavelength)
    if lam <= 0.0:
        raise ValueError("Wavelength must be strictly positive.")

    k_wave = 2.0 * np.pi / lam
    z_prime = lambda u: A * k_wave * np.cos(k_wave * u)
    integrand = lambda u: np.sqrt(1.0 + (z_prime(u)) ** 2)

    is_scalar = np.isscalar(x) or isinstance(x, (float, int))
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))

    # Arc length integration
    s_arr = np.zeros_like(x_arr)
    for idx, xi in enumerate(x_arr):
        s_val, _ = scipy.integrate.quad(integrand, 0.0, float(xi))
        s_arr[idx] = s_val

    # Curvature calculation
    zp = z_prime(x_arr)
    num = A * (k_wave ** 2) * np.abs(np.sin(k_wave * x_arr))
    denom = (1.0 + zp ** 2) ** 1.5
    kappa_arr = num / denom

    if is_scalar:
        return float(s_arr[0]), float(kappa_arr[0])
    return s_arr, kappa_arr


class SineCurvatureMapper:
    """
    1D Developable Sine Wave Curvature and Forming Compensation Engine.

    Maps unformed planar flat sheet positions (arc length s = X_flat) to local target
    re-entrant angles theta_0(s) and strut widths t_strut(s) to inversely compensate
    for outer fiber bending strain during thermoforming over z(x) = A * sin(2*pi*x / lambda).
    """

    def __init__(
        self,
        amplitude: float,
        wavelength: float,
        n_samples_per_period: int = 500,
    ) -> None:
        import scipy.integrate
        from scipy.interpolate import interp1d

        self.amplitude = float(amplitude)
        self.wavelength = float(wavelength)
        if self.wavelength <= 0.0:
            raise ValueError("Wavelength must be strictly positive.")

        self.k_wave = 2.0 * np.pi / self.wavelength
        self.kappa_max = float(self.amplitude * (self.k_wave ** 2))

        # Sample 1 full period [0, wavelength] for numerical arc length table
        x_dense = np.linspace(0.0, self.wavelength, n_samples_per_period + 1)
        integrand = lambda u: np.sqrt(1.0 + (self.amplitude * self.k_wave * np.cos(self.k_wave * u)) ** 2)
        s_dense = np.zeros_like(x_dense)
        for i in range(1, len(x_dense)):
            val, _ = scipy.integrate.quad(integrand, x_dense[i - 1], x_dense[i])
            s_dense[i] = s_dense[i - 1] + val

        self.S_period = float(s_dense[-1])
        self._s_from_x = interp1d(x_dense, s_dense, kind="cubic")
        self._x_from_s = interp1d(s_dense, x_dense, kind="cubic")

    def arc_length(self, x: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Compute arc length s(x) from projected coordinate x with periodic support."""
        is_scalar = np.isscalar(x) or isinstance(x, (float, int))
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        k = np.floor(x_arr / self.wavelength)
        x_rem = x_arr - k * self.wavelength
        s_val = k * self.S_period + self._s_from_x(np.clip(x_rem, 0.0, self.wavelength))
        return float(s_val[0]) if is_scalar else s_val

    def x_from_arc_length(self, s: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Invert arc length coordinate s = X_flat to projected coordinate x."""
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        k = np.floor(s_arr / self.S_period) if self.S_period > 0.0 else np.zeros_like(s_arr)
        s_rem = s_arr - k * self.S_period
        x_val = k * self.wavelength + self._x_from_s(np.clip(s_rem, 0.0, self.S_period))
        return float(x_val[0]) if is_scalar else x_val

    def curvature_at_x(self, x: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate local curvature kappa(x) in mm^-1."""
        is_scalar = np.isscalar(x) or isinstance(x, (float, int))
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        zp = self.amplitude * self.k_wave * np.cos(self.k_wave * x_arr)
        num = self.amplitude * (self.k_wave ** 2) * np.abs(np.sin(self.k_wave * x_arr))
        denom = (1.0 + zp ** 2) ** 1.5
        res = num / denom
        return float(res[0]) if is_scalar else res

    def curvature_at_s(self, s: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate local curvature kappa(s) at flat arc length coordinate s = X_flat."""
        x = self.x_from_arc_length(s)
        return self.curvature_at_x(x)

    def compensate_theta(
        self,
        s: float | Sequence[float] | np.ndarray,
        theta_nominal_deg: float = 20.0,
        sheet_thickness: float = 2.0,
        beta: float | None = None,
        theta_min_deg: float = 5.0,
        theta_max_deg: float = 45.0,
    ) -> float | np.ndarray:
        """
        Compute inversely compensated pre-closed re-entrant angle theta_0(s).

            theta_0(s) = theta^* - beta * (t_s / 2) * kappa(s)

        where beta is the kinematic sensitivity relating unit cell width expansion
        to internal angle change. Clamped above theta_min_deg to prevent strut locking.
        """
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        if beta is None:
            th_rad = np.radians(float(theta_nominal_deg))
            # beta_deg = (180 / pi) * [W_0 / (2*l*cos(theta_0))] ~ (180 / pi) / cos(theta_0)
            beta = (180.0 / np.pi) * (1.0 / max(np.cos(th_rad), 1e-4))

        kappa = np.asarray(self.curvature_at_s(s_arr), dtype=np.float64)
        eps_outer = kappa * (float(sheet_thickness) / 2.0)
        delta_theta = float(beta) * eps_outer
        theta_0 = float(theta_nominal_deg) - delta_theta
        theta_clamped = np.clip(theta_0, float(theta_min_deg), float(theta_max_deg))

        return float(theta_clamped[0]) if is_scalar else theta_clamped

    def compensate_strut_thickness(
        self,
        s: float | Sequence[float] | np.ndarray,
        t_base: float = 1.0,
        gamma: float = 0.15,
        max_scale: float = 3.0,
    ) -> float | np.ndarray:
        """
        Compute in-plane strut thickness grading t_strut(s) to modulate flexural rigidity:

            t_strut(s) = t_base * [1 + gamma * (kappa(s) / kappa_max)]
        """
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        kappa = np.asarray(self.curvature_at_s(s_arr), dtype=np.float64)
        ratio = kappa / max(self.kappa_max, 1e-12)
        t_mod = float(t_base) * (1.0 + float(gamma) * ratio)
        t_clamped = np.clip(t_mod, float(t_base), float(max_scale * t_base))

        return float(t_clamped[0]) if is_scalar else t_clamped


def eval_sine_compensated_theta(
    s: float | Sequence[float] | np.ndarray,
    amplitude: float,
    wavelength: float,
    theta_nominal_deg: float = 20.0,
    sheet_thickness: float = 2.0,
    beta: float | None = None,
    theta_min_deg: float = 5.0,
) -> float | np.ndarray:
    """Helper to evaluate compensated re-entrant angle theta_0(s) directly."""
    mapper = SineCurvatureMapper(amplitude, wavelength)
    return mapper.compensate_theta(
        s,
        theta_nominal_deg=theta_nominal_deg,
        sheet_thickness=sheet_thickness,
        beta=beta,
        theta_min_deg=theta_min_deg,
    )


def eval_sine_compensated_strut_width(
    s: float | Sequence[float] | np.ndarray,
    amplitude: float,
    wavelength: float,
    t_base: float = 1.0,
    gamma: float = 0.15,
) -> float | np.ndarray:
    """Helper to evaluate compensated strut width t_strut(s) directly."""
    mapper = SineCurvatureMapper(amplitude, wavelength)
    return mapper.compensate_strut_thickness(
        s,
        t_base=t_base,
        gamma=gamma,
    )


# =========================================================================
# General 1D Multi-Harmonic & Semi-Random Profile Curvature Engine
# =========================================================================

def compute_profile_curvature(
    x: float | Sequence[float] | np.ndarray,
    harmonics: Sequence[tuple[float, float, float]] | None = None,
    length: float = 140.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Compute developable arc length s(x) and local curvature kappa(x)
    for a general multi-harmonic or semi-random 1D surface profile.

    Parameters
    ----------
    x : float or array-like
        Projected coordinate(s) along X in mm.
    harmonics : list of (amplitude, wavelength, phase), optional
        Harmonic component tuples. If None, uses default semi-random profile.
    length : float
        Projected length of the profile domain in mm.
    """
    mapper = (
        ProfileCurvatureMapper.create_semirandom(length=length)
        if harmonics is None
        else ProfileCurvatureMapper(harmonics=harmonics, length=length)
    )
    s_val = mapper.arc_length(x)
    k_val = mapper.curvature_at_x(x)
    return s_val, k_val


class ProfileCurvatureMapper:
    """
    1D Developable Profile Curvature and Cell Size Compensation Engine.

    Supports arbitrary multi-harmonic, spline, or semi-random profiles z(x),
    computing:
    1. Arc length coordinate s(x) and total developable flat sheet length S_total
    2. Inverted developable flat mapping x(s)
    3. Local curvature kappa(x) and flat-coordinate curvature kappa(s)
    4. Curvature-driven continuous cell size distribution L(s) across flat sheet
    5. Discretized integer column boundaries [s_0, s_1, ..., s_M] guaranteeing whole-cell tessellation
    6. Inversely compensated re-entrant angles theta_0(s) and strut widths w(s)
    """

    def __init__(
        self,
        harmonics: Sequence[tuple[float, float, float]] | None = None,
        length: float = 140.0,
        z_func: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
        zp_func: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
        zpp_func: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
        n_samples: int = 1200,
    ) -> None:
        import scipy.integrate
        from scipy.interpolate import interp1d

        self.length = float(length)
        if self.length <= 0.0:
            raise ValueError("Profile length must be strictly positive.")

        self.harmonics = list(harmonics) if harmonics is not None else None
        self.x = np.linspace(0.0, self.length, n_samples)

        if self.harmonics is not None:
            self.z = np.zeros_like(self.x)
            self.zp = np.zeros_like(self.x)
            self.zpp = np.zeros_like(self.x)
            for A, lam, phi in self.harmonics:
                k = 2.0 * np.pi / float(lam)
                self.z += float(A) * np.sin(k * self.x + float(phi))
                self.zp += float(A) * k * np.cos(k * self.x + float(phi))
                self.zpp -= float(A) * (k**2) * np.sin(k * self.x + float(phi))
        elif z_func is not None:
            self.z = np.asarray(z_func(self.x), dtype=np.float64)
            if zp_func is not None:
                self.zp = np.asarray(zp_func(self.x), dtype=np.float64)
            else:
                self.zp = np.gradient(self.z, self.x)
            if zpp_func is not None:
                self.zpp = np.asarray(zpp_func(self.x), dtype=np.float64)
            else:
                self.zpp = np.gradient(self.zp, self.x)
        else:
            raise ValueError("Either harmonics or z_func must be provided.")

        # Compute numerical arc length coordinate s(x)
        speed = np.sqrt(1.0 + self.zp**2)
        s_arr = scipy.integrate.cumulative_trapezoid(speed, self.x, initial=0.0)
        self.s = s_arr
        self.S_total = float(s_arr[-1])

        # Curvature along projected coordinate x
        num = np.abs(self.zpp)
        denom = (1.0 + self.zp**2) ** 1.5
        self.kappa_x = num / np.maximum(denom, 1e-12)
        self.kappa_max = float(np.max(self.kappa_x))
        self.kappa_min = float(np.min(self.kappa_x))

        # Continuous invertible interpolators between s and x
        self._s_from_x = interp1d(self.x, self.s, kind="cubic", fill_value="extrapolate")
        self._x_from_s = interp1d(self.s, self.x, kind="cubic", fill_value="extrapolate")

    @classmethod
    def create_semirandom(
        cls,
        length: float = 140.0,
        seed: int = 42,
        n_harmonics: int = 3,
        amplitude_scale: float = 1.0,
    ) -> ProfileCurvatureMapper:
        """
        Create a semi-random 1D developable profile with rich asymmetric curvature features.

        Features include sharp localized bends (high curvature), gentle broad transitions,
        and flat inflection zones.
        """
        if seed == 42 and n_harmonics == 3:
            # Calibrated default: high curvature bend, inflection zone, broad crest
            harmonics = [
                (18.0 * amplitude_scale, 140.0, 0.0),
                (10.0 * amplitude_scale, 50.0, 1.0),
                (4.0 * amplitude_scale, 24.0, 2.2),
            ]
        else:
            rng = np.random.default_rng(seed)
            harmonics = []
            # 1. Macro wave
            a1 = float(rng.uniform(0.10, 0.15) * length * amplitude_scale)
            harmonics.append((a1, float(length), 0.0))
            # 2. Secondary intermediate wave
            a2 = float(rng.uniform(0.05, 0.08) * length * amplitude_scale)
            l2 = float(rng.uniform(0.30, 0.45) * length)
            phi2 = float(rng.uniform(0.5, 2.5))
            harmonics.append((a2, l2, phi2))
            # 3. Fine ripple / sharp bend
            for _ in range(max(1, n_harmonics - 2)):
                a3 = float(rng.uniform(0.02, 0.04) * length * amplitude_scale)
                l3 = float(rng.uniform(0.14, 0.22) * length)
                phi3 = float(rng.uniform(0.0, 2.0 * np.pi))
                harmonics.append((a3, l3, phi3))

        return cls(harmonics=harmonics, length=length)

    def arc_length(self, x: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate arc length s(x) from projected coordinate x in mm."""
        is_scalar = np.isscalar(x) or isinstance(x, (float, int))
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        s_val = self._s_from_x(np.clip(x_arr, 0.0, self.length))
        return float(s_val[0]) if is_scalar else s_val

    def x_from_arc_length(self, s: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Invert flat arc length coordinate s to projected coordinate x in mm."""
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        x_val = self._x_from_s(np.clip(s_arr, 0.0, self.S_total))
        return float(x_val[0]) if is_scalar else x_val

    def z_at_x(self, x: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate profile elevation z(x) in mm."""
        is_scalar = np.isscalar(x) or isinstance(x, (float, int))
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        res = np.interp(np.clip(x_arr, 0.0, self.length), self.x, self.z)
        return float(res[0]) if is_scalar else res

    def curvature_at_x(self, x: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate local curvature kappa(x) in mm^-1."""
        is_scalar = np.isscalar(x) or isinstance(x, (float, int))
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        res = np.interp(np.clip(x_arr, 0.0, self.length), self.x, self.kappa_x)
        return float(res[0]) if is_scalar else res

    def curvature_at_s(self, s: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate local curvature kappa(s) along unformed flat arc length coordinate s = X_flat."""
        x = self.x_from_arc_length(s)
        return self.curvature_at_x(x)

    def cell_size_at_s(
        self,
        s: float | Sequence[float] | np.ndarray,
        cell_size_min: float = 3.5,
        cell_size_max: float = 14.0,
        power: float = 0.5,
    ) -> float | np.ndarray:
        """
        Evaluate curvature-compensated unit cell size L(s) along flat sheet coordinate s.

        High curvature regions (sharp bends) receive fine, compliant cells L_min,
        while low curvature / inflection regions receive large cells L_max.

            L(s) = L_max - (L_max - L_min) * (kappa(s) / kappa_max)^power
        """
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        kappa = np.asarray(self.curvature_at_s(s_arr), dtype=np.float64)
        norm = np.clip(kappa / max(self.kappa_max, 1e-12), 0.0, 1.0)
        l_min = float(cell_size_min)
        l_max = float(cell_size_max)
        l_val = l_max - (l_max - l_min) * (norm ** float(power))
        l_clamped = np.clip(l_val, l_min, l_max)
        return float(l_clamped[0]) if is_scalar else l_clamped

    def compute_column_boundaries(
        self,
        cell_size_min: float = 3.5,
        cell_size_max: float = 14.0,
        power: float = 0.5,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Compute discrete whole-cell column boundaries [s_0, s_1, ..., s_M] along [0, S_total].

        Uses cumulative density inversion N(s) = \\int_0^s 1/L(u) du to ensure
        integer whole cells with local width a_i ≈ L(s_i).
        """
        import scipy.integrate
        from scipy.interpolate import interp1d

        s_dense = np.linspace(0.0, self.S_total, n_samples)
        l_dense = np.asarray(self.cell_size_at_s(s_dense, cell_size_min, cell_size_max, power), dtype=np.float64)
        inv_l = 1.0 / np.maximum(l_dense, 1e-6)

        n_cum = scipy.integrate.cumulative_trapezoid(inv_l, s_dense, initial=0.0)
        n_total = float(n_cum[-1])
        m_cols = max(2, int(np.round(n_total)))

        n_norm = (float(m_cols) / max(n_total, 1e-12)) * n_cum
        cum_to_s = interp1d(n_norm, s_dense, kind="cubic", fill_value="extrapolate")

        bounds = cum_to_s(np.arange(m_cols + 1, dtype=np.float64))
        bounds[0] = 0.0
        bounds[-1] = self.S_total
        return bounds

    def compensate_theta(
        self,
        s: float | Sequence[float] | np.ndarray,
        theta_nominal_deg: float = 20.0,
        sheet_thickness: float = 2.0,
        beta: float | None = None,
        theta_min_deg: float = 5.0,
        theta_max_deg: float = 45.0,
    ) -> float | np.ndarray:
        """
        Compute inversely compensated pre-closed re-entrant angle theta_0(s).

            theta_0(s) = theta^* - beta * (t_s / 2) * kappa(s)
        """
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        if beta is None:
            th_rad = np.radians(float(theta_nominal_deg))
            beta = (180.0 / np.pi) * (1.0 / max(np.cos(th_rad), 1e-4))

        kappa = np.asarray(self.curvature_at_s(s_arr), dtype=np.float64)
        eps_outer = kappa * (float(sheet_thickness) / 2.0)
        delta_theta = float(beta) * eps_outer
        theta_0 = float(theta_nominal_deg) - delta_theta
        theta_clamped = np.clip(theta_0, float(theta_min_deg), float(theta_max_deg))

        return float(theta_clamped[0]) if is_scalar else theta_clamped

    def compensate_strut_thickness(
        self,
        s: float | Sequence[float] | np.ndarray,
        t_base: float = 1.0,
        gamma: float = 0.15,
        max_scale: float = 3.0,
    ) -> float | np.ndarray:
        """
        Compute in-plane strut thickness grading t_strut(s) to modulate flexural rigidity:

            t_strut(s) = t_base * [1 + gamma * (kappa(s) / kappa_max)]
        """
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        kappa = np.asarray(self.curvature_at_s(s_arr), dtype=np.float64)
        ratio = kappa / max(self.kappa_max, 1e-12)
        t_mod = float(t_base) * (1.0 + float(gamma) * ratio)
        t_clamped = np.clip(t_mod, float(t_base), float(max_scale * t_base))

        return float(t_clamped[0]) if is_scalar else t_clamped

    def constant_solid_fraction_strut_width(
        self,
        s: float | Sequence[float] | np.ndarray,
        cell_size_min: float = 3.5,
        cell_size_max: float = 14.0,
        power: float = 0.5,
        solid_fraction: float = 0.15,
        topology: str = "square",
    ) -> float | np.ndarray:
        """
        Compute strut width w(s) maintaining constant solid fraction phi_0 as L(s) varies.
        """
        is_scalar = np.isscalar(s) or isinstance(s, (float, int))
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        l_arr = np.asarray(self.cell_size_at_s(s_arr, cell_size_min, cell_size_max, power), dtype=np.float64)
        w_arr = np.array([
            width_for_constant_solid_fraction(l_val, solid_fraction=solid_fraction, topology=topology)
            for l_val in l_arr
        ], dtype=np.float64)
        return float(w_arr[0]) if is_scalar else w_arr


def solve_reentrant_dimensions_from_pore(
    target_pore_mm: float | None = None,
    wall_thickness_mm: float = 0.4,
    mode: str = "mic",
    w_top_ratio: float = 0.65,
    w_waist_ratio: float = 0.30,
    aspect_ratio_b_over_a: float = 1.0,
    *,
    target_mic_pore_mm: float | None = None,
) -> dict[str, float]:
    """
    Solve unit cell pitch a and height b required to achieve a target
    Maximal Inscribed Circle (MIC) clear pore opening diameter or throat constriction.

    Parameters
    ----------
    target_pore_mm : float
        Target clear pore diameter in mm (e.g. 1.8 mm for tissue ingrowth).
    wall_thickness_mm : float
        In-plane strut thickness in mm.
    mode : str
        'mic' (default): Solves for the true Maximal Inscribed Circle diameter inside the void chamber.
            In standard re-entrant cells (w_top=0.65, w_waist=0.30, b=a), D_mic ≈ 0.47389*a - t_strut.
        'throat': Solves for the bottleneck waist constriction width (w_waist - t_strut).
    w_top_ratio : float
        Cap width ratio relative to pitch a (default 0.65).
    w_waist_ratio : float
        Waist width ratio relative to pitch a (default 0.30).
    aspect_ratio_b_over_a : float
        Cell height-to-width ratio b / a (default 1.0).
    target_mic_pore_mm : float, optional
        Backwards-compatible alias for target_pore_mm.

    Returns
    -------
    dict[str, float]
        Dictionary with 'a_pitch_mm', 'b_height_mm', 'w_top_mm', 'w_waist_mm',
        'estimated_mic_pore_mm', and 'estimated_throat_pore_mm'.
    """
    if target_pore_mm is None:
        if target_mic_pore_mm is not None:
            target_pore_mm = target_mic_pore_mm
        else:
            raise ValueError("target_pore_mm (or target_mic_pore_mm) must be specified.")

    mode_lower = mode.lower().strip()
    target_val = float(target_pore_mm)
    wall_t = float(wall_thickness_mm)

    if mode_lower in ("mic", "maximal_inscribed_circle", "chamber"):
        # Calibrated geometric invariance: D_MIC ≈ 0.4738886 * a - wall_thickness
        alpha_mic = 0.47388857 * (w_waist_ratio / 0.30)
        a_pitch = (target_val + wall_t) / alpha_mic
    elif mode_lower in ("throat", "waist", "constriction"):
        if w_waist_ratio <= 0.0:
            raise ValueError(f"w_waist_ratio must be positive, got {w_waist_ratio}")
        a_pitch = (target_val + wall_t) / float(w_waist_ratio)
    else:
        raise ValueError(f"Unsupported pore solving mode '{mode}'. Expected 'mic' or 'throat'.")

    b_height = a_pitch * float(aspect_ratio_b_over_a)
    w_top = w_top_ratio * a_pitch
    w_waist = w_waist_ratio * a_pitch

    est_mic = 0.47388857 * a_pitch - wall_t
    est_throat = w_waist - wall_t

    return {
        "a_pitch_mm": float(a_pitch),
        "b_height_mm": float(b_height),
        "w_top_mm": float(w_top),
        "w_waist_mm": float(w_waist),
        "target_pore_mm": target_val,
        "wall_thickness_mm": wall_t,
        "estimated_mic_pore_mm": float(est_mic),
        "estimated_throat_pore_mm": float(est_throat),
    }


def analyze_pore_maximal_inscribed_circles(
    domain_width: float,
    domain_height: float,
    n_circumferential: int,
    strut_w: float | np.ndarray | Sequence[float],
    periodic_x: bool = False,
    tolerance: float = 0.001,
) -> dict:
    """
    Compute Maximal Inscribed Circles (MIC) for all interior void pores
    in a re-entrant planar lattice using the pole of inaccessibility (polylabel).

    Parameters
    ----------
    domain_width : float
        Domain width in mm.
    domain_height : float
        Domain height in mm.
    n_circumferential : int
        Number of unit cells along width.
    strut_w : float or array
        Strut width in mm.
    periodic_x : bool
        If True, enforces periodic boundary.
    tolerance : float
        Precision tolerance for polylabel search.

    Returns
    -------
    dict
        Dictionary containing pore count, MIC diameters, centers,
        mean diameter, min diameter, max diameter, and solid fraction.
    """
    from shapely.geometry import box, LineString, Polygon, MultiPolygon
    from shapely.ops import unary_union, polylabel
    from graphite.explicit.surface_lattice.unit_cells import tessellate_reentrant_domain

    segs, meta = tessellate_reentrant_domain(
        domain_width=domain_width,
        domain_height=domain_height,
        n_circumferential=n_circumferential,
        periodic_x=periodic_x,
    )

    domain_box = box(0.0, 0.0, domain_width, domain_height)

    # Thicken struts into polygons
    is_seq = isinstance(strut_w, (list, tuple, np.ndarray))
    strut_polys = []
    for idx, (p1, p2) in enumerate(segs):
        w = float(strut_w[idx]) if is_seq else float(strut_w)
        line = LineString([p1, p2])
        strut_polys.append(line.buffer(w / 2.0, cap_style=2))

    solid = unary_union(strut_polys)
    solid_clipped = solid.intersection(domain_box)
    void = domain_box.difference(solid_clipped)

    solid_fraction = solid_clipped.area / domain_box.area

    # Extract all complete interior pores (not clipped by outer boundary)
    geoms = void.geoms if isinstance(void, MultiPolygon) else [void]
    interior_pores: list[Polygon] = []
    for g in geoms:
        if not g.boundary.intersects(domain_box.boundary):
            interior_pores.append(g)

    mic_results = []
    for p in interior_pores:
        center_pt = polylabel(p, tolerance=tolerance)
        r_mic = p.boundary.distance(center_pt)
        d_mic = 2.0 * r_mic
        mic_results.append({
            "center": (float(center_pt.x), float(center_pt.y)),
            "radius": float(r_mic),
            "diameter": float(d_mic),
            "area": float(p.area),
        })

    diameters = [m["diameter"] for m in mic_results]
    return {
        "n_pores": len(mic_results),
        "mean_d_mic": float(np.mean(diameters)) if diameters else 0.0,
        "min_d_mic": float(np.min(diameters)) if diameters else 0.0,
        "max_d_mic": float(np.max(diameters)) if diameters else 0.0,
        "std_d_mic": float(np.std(diameters)) if diameters else 0.0,
        "diameters": diameters,
        "solid_fraction": float(solid_fraction),
        "pore_details": mic_results,
    }

