"""
Sparse direct linear solvers for Aristo FEA (K u = F).

Supports SciPy SuperLU/UMFPACK (default fallback) and Intel MKL PARDISO via pypardiso.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve as scipy_spsolve

if TYPE_CHECKING:
    from graphite.aristo.aristo_config import AristoConfig

_PARDISO_AVAILABLE: bool | None = None


def pypardiso_available() -> bool:
    """Return True when ``pypardiso`` (MKL PARDISO) can be imported."""
    global _PARDISO_AVAILABLE
    if _PARDISO_AVAILABLE is None:
        try:
            import pypardiso  # noqa: F401

            _PARDISO_AVAILABLE = True
        except ImportError:
            _PARDISO_AVAILABLE = False
    return _PARDISO_AVAILABLE


def resolve_linear_solver_key(config: AristoConfig | None = None) -> str:
    """
    Resolve solver backend: ``scipy`` or ``pardiso``.

    Config ``fea_linear_solver`` overrides env ``ARISTO_LINEAR_SOLVER``.
    ``auto`` (default) picks PARDISO when installed, else SciPy.
    """
    if config is not None and getattr(config, "fea_linear_solver", None):
        key = str(config.fea_linear_solver).strip().lower()
    else:
        key = os.environ.get("ARISTO_LINEAR_SOLVER", "auto").strip().lower()

    if key in ("auto", "", "default"):
        return "pardiso" if pypardiso_available() else "scipy"
    if key in ("scipy", "superlu", "umfpack"):
        return "scipy"
    if key in ("pardiso", "pypardiso", "mkl"):
        return "pardiso"
    raise ValueError(
        f"fea_linear_solver / ARISTO_LINEAR_SOLVER must be auto, scipy, or pardiso; got {key!r}."
    )


def get_sparse_solve_fn(key: str) -> tuple[Callable[..., np.ndarray], str]:
    """Return ``(solve_fn, backend_label)`` for a resolved key."""
    if key == "pardiso":
        if not pypardiso_available():
            raise RuntimeError(
                "fea_linear_solver='pardiso' but pypardiso is not installed. "
                "Install with: pip install pypardiso"
            )
        import pypardiso

        return pypardiso.spsolve, "pypardiso"
    return scipy_spsolve, "scipy"


def sparse_direct_solve(
    K: spmatrix,
    F: np.ndarray,
    config: AristoConfig | None = None,
) -> tuple[np.ndarray, str]:
    """
    Solve ``K @ u = F`` with the configured sparse direct backend.

    Returns ``(u, backend_label)`` where ``backend_label`` is ``scipy`` or ``pypardiso``.
    """
    key = resolve_linear_solver_key(config)
    solve_fn, backend = get_sparse_solve_fn(key)
    u = solve_fn(K.tocsc(), np.asarray(F, dtype=np.float64).ravel())
    return np.asarray(u, dtype=np.float64).ravel(), backend
