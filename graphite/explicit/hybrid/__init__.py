"""
Graphite Explicit — Multilayer & Hybrid Lattice Optimization Engine

Based on Liu et al. (Nature Communications 2024):
"Ultrastiff metamaterials generated through a multilayer strategy and topology optimization"

Provides:
- Level-Set ODE-driven density methods and regularization.
- Regularized Heaviside mapping H_eps(phi) and derivative h_eps(phi).
- Mean curvature H(phi) computation.
- Continuous multi-morphology interface interpolation suppressing parasitic interface bending and stress singularities.
"""

from __future__ import annotations

from .levelset import (
    regularized_heaviside,
    heaviside_derivative,
    levelset_regularization_step,
    mean_curvature_field,
    blend_lattice_morphologies,
    generate_hybrid_levelset_lattice,
)

__all__ = [
    "regularized_heaviside",
    "heaviside_derivative",
    "levelset_regularization_step",
    "mean_curvature_field",
    "blend_lattice_morphologies",
    "generate_hybrid_levelset_lattice",
]
