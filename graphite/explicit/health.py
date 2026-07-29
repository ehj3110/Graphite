"""Dependency health checks for the explicit lattice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util
from typing import Literal

StageName = Literal["topology", "scaffold", "geometry", "full"]

_STAGE_MODULES: dict[StageName, tuple[str, ...]] = {
    "topology": ("numpy", "scipy"),
    "scaffold": ("gmsh", "numpy", "trimesh"),
    "geometry": ("manifold3d", "numpy", "trimesh"),
    "full": ("numpy", "scipy", "gmsh", "trimesh", "manifold3d"),
}


@dataclass(frozen=True)
class DependencyStatus:
    module: str
    present: bool
    version: str | None


def _get_version(module_name: str) -> str | None:
    try:
        return metadata.version(module_name)
    except metadata.PackageNotFoundError:
        return None


def _validate_stage(stage: str) -> StageName:
    if stage not in _STAGE_MODULES:
        valid = ", ".join(_STAGE_MODULES.keys())
        raise ValueError(f"Unknown stage '{stage}'. Supported: {valid}.")
    return stage  # type: ignore[return-value]


def check_explicit_health(stage: str = "full") -> dict[str, DependencyStatus]:
    """Return dependency availability for an explicit pipeline stage."""
    stage_name = _validate_stage(stage)
    report: dict[str, DependencyStatus] = {}
    for module_name in _STAGE_MODULES[stage_name]:
        present = util.find_spec(module_name) is not None
        report[module_name] = DependencyStatus(
            module=module_name,
            present=present,
            version=_get_version(module_name) if present else None,
        )
    return report


def missing_dependencies(stage: str = "full") -> list[str]:
    """Return a list of missing modules for the selected stage."""
    report = check_explicit_health(stage)
    return [name for name, status in report.items() if not status.present]


def require_explicit_dependencies(stage: str = "full") -> None:
    """Raise a helpful error if dependencies for stage are missing."""
    missing = missing_dependencies(stage)
    if not missing:
        return
    missing_csv = ", ".join(missing)
    raise RuntimeError(
        f"Missing dependencies for explicit '{stage}' stage: {missing_csv}. "
        "Install the missing packages in your active Python environment."
    )
