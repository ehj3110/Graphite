"""
Graphite Math - TPMS Equations

This module contains the mathematical definitions and evaluation functions for 
Triply Periodic Minimal Surfaces (TPMS). It supports direct spatial evaluation 
as well as decoupled phase evaluation for use in functionally graded or chirped 
lattice generation.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid


def gyroid(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True):
    """
    Evaluate the Gyroid TPMS equation.

    Parameters
    ----------
    x, y, z : ndarray
        The spatial coordinates to evaluate.
    unit_cell_size : float
        The physical size of the unit cell bounding box.
    iso_offset : float, optional
        Isovalue offset (thickness control), by default 0.0.
    is_sheet : bool, optional
        If True, evaluates as a sheet network (absolute magnitude). If False, 
        evaluates as a solid/void level-set. By default True.

    Returns
    -------
    ndarray
        The evaluated scalar field.
    """
    k = (2 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    eq = np.sin(kx) * np.cos(ky) + np.sin(ky) * np.cos(kz) + np.sin(kz) * np.cos(kx)
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset


def schwarz_p(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True):
    """
    Evaluate the Schwarz Primitive (P) TPMS equation.

    Parameters
    ----------
    x, y, z : ndarray
        The spatial coordinates to evaluate.
    unit_cell_size : float
        The physical size of the unit cell bounding box.
    iso_offset : float, optional
        Isovalue offset (thickness control), by default 0.0.
    is_sheet : bool, optional
        If True, evaluates as a sheet network. By default True.

    Returns
    -------
    ndarray
        The evaluated scalar field.
    """
    k = (2 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    eq = np.cos(kx) + np.cos(ky) + np.cos(kz)
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset


def schwarz_d(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True):
    """
    Evaluate the Schwarz Diamond (D) TPMS equation.

    Parameters
    ----------
    x, y, z : ndarray
        The spatial coordinates to evaluate.
    unit_cell_size : float
        The physical size of the unit cell bounding box.
    iso_offset : float, optional
        Isovalue offset (thickness control), by default 0.0.
    is_sheet : bool, optional
        If True, evaluates as a sheet network. By default True.

    Returns
    -------
    ndarray
        The evaluated scalar field.
    """
    k = (2 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    eq = (
        np.sin(kx) * np.sin(ky) * np.sin(kz)
        + np.sin(kx) * np.cos(ky) * np.cos(kz)
        + np.cos(kx) * np.sin(ky) * np.cos(kz)
        + np.cos(kx) * np.cos(ky) * np.sin(kz)
    )
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset


def schwarz_diamond(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True):
    """Schwarz Diamond variant: cos(x)cos(y)cos(z) - sin(x)sin(y)sin(z)."""
    k = (2 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    eq = np.cos(kx) * np.cos(ky) * np.cos(kz) - np.sin(kx) * np.sin(ky) * np.sin(kz)
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset


def lidinoid(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True):
    """
    Evaluate the Lidinoid TPMS equation.

    Parameters
    ----------
    x, y, z : ndarray
        The spatial coordinates to evaluate.
    unit_cell_size : float
        The physical size of the unit cell bounding box.
    iso_offset : float, optional
        Isovalue offset (thickness control), by default 0.0.
    is_sheet : bool, optional
        If True, evaluates as a sheet network. By default True.

    Returns
    -------
    ndarray
        The evaluated scalar field.
    """
    k = (2 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    eq = (
        np.sin(2 * kx) * np.cos(ky) * np.sin(kz)
        + np.sin(2 * ky) * np.cos(kz) * np.sin(kx)
        + np.sin(2 * kz) * np.cos(kx) * np.sin(ky)
        - np.cos(2 * kx) * np.cos(2 * ky)
        - np.cos(2 * ky) * np.cos(2 * kz)
        - np.cos(2 * kz) * np.cos(2 * kx)
        + 0.3
    )
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset


def split_p(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True):
    """
    Evaluate the Split-P TPMS equation.

    Parameters
    ----------
    x, y, z : ndarray
        The spatial coordinates to evaluate.
    unit_cell_size : float
        The physical size of the unit cell bounding box.
    iso_offset : float, optional
        Isovalue offset (thickness control), by default 0.0.
    is_sheet : bool, optional
        If True, evaluates as a sheet network. By default True.

    Returns
    -------
    ndarray
        The evaluated scalar field.
    """
    k = (2 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    t1 = (
        np.sin(2 * kx) * np.sin(kz) * np.cos(ky)
        + np.sin(2 * ky) * np.sin(kx) * np.cos(kz)
        + np.sin(2 * kz) * np.sin(ky) * np.cos(kx)
    )
    t2 = (
        np.cos(2 * kx) * np.cos(2 * ky)
        + np.cos(2 * ky) * np.cos(2 * kz)
        + np.cos(2 * kz) * np.cos(2 * kx)
    )
    t3 = np.cos(2 * kx) + np.cos(2 * ky) + np.cos(2 * kz)
    eq = 1.1 * t1 - 0.2 * t2 - 0.4 * t3
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset


def neovius(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True):
    """
    Evaluate the Neovius TPMS equation.

    Parameters
    ----------
    x, y, z : ndarray
        The spatial coordinates to evaluate.
    unit_cell_size : float
        The physical size of the unit cell bounding box.
    iso_offset : float, optional
        Isovalue offset (thickness control), by default 0.0.
    is_sheet : bool, optional
        If True, evaluates as a sheet network. By default True.

    Returns
    -------
    ndarray
        The evaluated scalar field.
    """
    k = (2 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    eq = 3.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz)) + 4.0 * (
        np.cos(kx) * np.cos(ky) * np.cos(kz)
    )
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset


def evaluate_tpms(lattice_type, k, X, Y, Z):
    """
    Evaluate a TPMS field for the given lattice type.

    Parameters
    ----------
    lattice_type : str
        The name of the TPMS equation to evaluate (e.g., 'gyroid', 'diamond', 
        'schwarz primitive', 'lidinoid', 'neovius', 'split-p').
    k : float
        The spatial frequency (2 * pi / unit_cell_size).
    X : ndarray
        The X-coordinate grid.
    Y : ndarray
        The Y-coordinate grid.
    Z : ndarray
        The Z-coordinate grid.

    Returns
    -------
    ndarray
        The evaluated TPMS scalar field (f(X, Y, Z)) as sheet networks 
        (magnitude evaluation).
    """
    l_type = lattice_type.lower()
    unit_cell_size = (2 * np.pi) / k
    
    if l_type == "gyroid":
        return gyroid(X, Y, Z, unit_cell_size, iso_offset=0.0, is_sheet=False)
    elif l_type in ["schwarz-p", "schwarz primitive", "schwarz"]:
        return schwarz_p(X, Y, Z, unit_cell_size, iso_offset=0.0, is_sheet=False)
    elif l_type == "diamond":
        return schwarz_d(X, Y, Z, unit_cell_size, iso_offset=0.0, is_sheet=False)
    elif l_type in ["schwarz-diamond", "schwarz diamond", "schwarz-d"]:
        return schwarz_diamond(X, Y, Z, unit_cell_size, iso_offset=0.0, is_sheet=False)
    elif l_type == "neovius":
        return neovius(X, Y, Z, unit_cell_size, iso_offset=0.0, is_sheet=False)
    elif l_type == "lidinoid":
        return lidinoid(X, Y, Z, unit_cell_size, iso_offset=0.0, is_sheet=False)
    elif l_type in ["split-p", "split_p"]:
        return split_p(X, Y, Z, unit_cell_size, iso_offset=0.0, is_sheet=False)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def evaluate_tpms_phase(lattice_type, U, V, W):
    """
    Evaluate a TPMS field from precomputed phase coordinates (U, V, W).

    This allows decoupling the geometric frequency/period grading from the 
    actual surface evaluation. Useful for graded, chirped, or conformal fields 
    where U, V, and W are non-linear or derived from integrated phase.

    Parameters
    ----------
    lattice_type : str
        The name of the TPMS equation to evaluate.
    U : ndarray
        The integrated phase in the X (or primary) direction.
    V : ndarray
        The integrated phase in the Y (or secondary) direction.
    W : ndarray
        The integrated phase in the Z (or tertiary) direction.

    Returns
    -------
    ndarray
        The evaluated unrectified TPMS scalar field (f(U, V, W)).
    """
    l_type = lattice_type.lower()
    if l_type == "gyroid":
        return (
            np.sin(U) * np.cos(V)
            + np.sin(V) * np.cos(W)
            + np.sin(W) * np.cos(U)
        )
    elif l_type in ["schwarz-p", "schwarz primitive", "schwarz"]:
        return np.cos(U) + np.cos(V) + np.cos(W)
    elif l_type == "diamond":
        return (
            np.sin(U) * np.sin(V) * np.sin(W)
            + np.sin(U) * np.cos(V) * np.cos(W)
            + np.cos(U) * np.sin(V) * np.cos(W)
            + np.cos(U) * np.cos(V) * np.sin(W)
        )
    elif l_type in ["schwarz-diamond", "schwarz diamond", "schwarz-d"]:
        return np.cos(U) * np.cos(V) * np.cos(W) - np.sin(U) * np.sin(V) * np.sin(W)
    elif l_type == "neovius":
        return 3.0 * (np.cos(U) + np.cos(V) + np.cos(W)) + 4.0 * (
            np.cos(U) * np.cos(V) * np.cos(W)
        )
    elif l_type == "lidinoid":
        return (
            np.sin(2 * U) * np.cos(V) * np.sin(W)
            + np.sin(2 * V) * np.cos(W) * np.sin(U)
            + np.sin(2 * W) * np.cos(U) * np.sin(V)
            - np.cos(2 * U) * np.cos(2 * V)
            - np.cos(2 * V) * np.cos(2 * W)
            - np.cos(2 * W) * np.cos(2 * U)
            + 0.3
        )
    elif l_type in ["split-p", "split_p"]:
        t1 = (
            np.sin(2 * U) * np.sin(W) * np.cos(V)
            + np.sin(2 * V) * np.sin(U) * np.cos(W)
            + np.sin(2 * W) * np.sin(V) * np.cos(U)
        )
        t2 = (
            np.cos(2 * U) * np.cos(2 * V)
            + np.cos(2 * V) * np.cos(2 * W)
            + np.cos(2 * W) * np.cos(2 * U)
        )
        t3 = np.cos(2 * U) + np.cos(2 * V) + np.cos(2 * W)
        return 1.1 * t1 - 0.2 * t2 - 0.4 * t3
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def calculate_integrated_phase(distances, control_points, control_L):
    """
    Integrate local TPMS frequency along a 1D distance axis.

    Parameters
    ----------
    distances : 1D array
        Axis samples where phase should be computed.
    control_points : 1D array-like
        Distance knot positions.
    control_L : 1D array-like
        Target pore sizes at each knot.

    Returns
    -------
    phase : 1D ndarray
        Integrated phase W(distance) sampled at `distances`.
    """
    d = np.asarray(distances, dtype=float).ravel()
    if d.size == 0:
        return d

    cp = np.asarray(control_points, dtype=float).ravel()
    lvals = np.asarray(control_L, dtype=float).ravel()
    if cp.size != lvals.size:
        raise ValueError("control_points and control_L must have the same length")
    if cp.size < 2:
        raise ValueError("control_points must contain at least two values")

    order = np.argsort(cp)
    cp = cp[order]
    lvals = lvals[order]
    lvals = np.maximum(lvals, 1e-6)

    sort_idx = np.argsort(d)
    d_sorted = d[sort_idx]
    l_sorted = np.interp(d_sorted, cp, lvals)
    omega_sorted = 2.0 * np.pi / np.maximum(l_sorted, 1e-6)
    phase_sorted = cumulative_trapezoid(omega_sorted, d_sorted, initial=0.0)

    phase = np.empty_like(phase_sorted)
    phase[sort_idx] = phase_sorted
    return phase

