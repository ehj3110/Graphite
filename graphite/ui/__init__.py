"""
Graphite UI — Interactive Trame + PyVista engineering frontend for conformal lattices.
"""

from .surface_preview import generate_surface_tpms_preview, generate_auxetic_preview
from .cli import run_headless_recipe

__all__ = ["generate_surface_tpms_preview", "generate_auxetic_preview", "run_headless_recipe"]
