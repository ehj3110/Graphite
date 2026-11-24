"""
Graphite UI — Native PySide6 + VTK Implicit Studio & legacy engineering frontends.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ImplicitStudioState",
    "ImplicitStudioWindow",
    "launch_implicit_studio",
    "generate_surface_tpms_preview",
    "generate_auxetic_preview",
    "run_headless_recipe",
    "UsdRuleLabWindow",
    "launch_usd_rule_lab",
]


def __getattr__(name: str) -> Any:
    if name == "ImplicitStudioState":
        from .implicit_state import ImplicitStudioState

        return ImplicitStudioState
    if name in ("ImplicitStudioWindow", "launch_implicit_studio"):
        from . import implicit_studio

        return getattr(implicit_studio, name)
    if name in ("generate_surface_tpms_preview", "generate_auxetic_preview"):
        from . import surface_preview

        return getattr(surface_preview, name)
    if name == "run_headless_recipe":
        from .cli import run_headless_recipe

        return run_headless_recipe
    if name in ("UsdRuleLabWindow", "launch_usd_rule_lab"):
        from . import usd_rule_lab

        return getattr(usd_rule_lab, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
