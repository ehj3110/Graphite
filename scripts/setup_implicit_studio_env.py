#!/usr/bin/env python3
"""Lab workstation environment checker & bootstrapper for Graphite Implicit Studio.

Usage:
    python scripts/setup_implicit_studio_env.py --check
    python scripts/setup_implicit_studio_env.py --create-venv .venv_studio
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys

REQUIRED_MODULES: tuple[tuple[str, str], ...] = (
    ("numpy", "NumPy arrays"),
    ("scipy", "EDT / SciPy optimization"),
    ("trimesh", "STL / triangle mesh IO"),
    ("manifold3d", "Exact manifold CSG booleans"),
    ("PySide6", "Desktop Qt6 UI"),
    ("vtk", "3D OpenGL viewport & Flying Edges"),
    ("PIL", "Pillow 2D slice & snapshot capture"),
    ("networkx", "Micropillar & surface region graph"),
)

OPTIONAL_MODULES: tuple[tuple[str, str], ...] = (
    ("lxml", "Optional native .3MF export"),
    ("yaml", "Optional YAML headless recipes"),
)


def check_environment() -> tuple[bool, list[str]]:
    """Verify runtime dependencies for Graphite Implicit Studio."""
    lines: list[str] = [f"Python: {sys.version.split()[0]} ({sys.executable})"]
    all_ok = sys.version_info >= (3, 10)
    if not all_ok:
        lines.append("FAIL python>=3.10 required")

    for mod_name, desc in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "")
            lines.append(f"OK   {mod_name:12s} {ver:10s} ({desc})")
        except Exception as exc:
            all_ok = False
            lines.append(f"MISS {mod_name:12s} {'':10s} ({desc}) -> {exc}")

    for mod_name, desc in OPTIONAL_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "")
            lines.append(f"OPT  {mod_name:12s} {ver:10s} ({desc})")
        except Exception:
            lines.append(f"WARN {mod_name:12s} {'not installed':13s} ({desc})")

    return all_ok, lines


def create_venv(venv_dir: Path, *, install_lxml: bool = False) -> int:
    """Create a dedicated lab workstation virtual environment and install dependencies."""
    repo_root = Path(__file__).resolve().parents[1]
    req_file = repo_root / "requirements-implicit-studio.txt"
    print(f"Creating virtual environment at: {venv_dir}")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    py_bin = (
        venv_dir / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv_dir / "bin" / "python"
    )
    subprocess.check_call([str(py_bin), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(py_bin), "-m", "pip", "install", "-r", str(req_file)])
    if install_lxml:
        subprocess.check_call([str(py_bin), "-m", "pip", "install", "lxml>=5.0"])
    print(f"Done. Verify with:\n  {py_bin} scripts/setup_implicit_studio_env.py --check")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Graphite Implicit Studio Lab Setup & Check")
    parser.add_argument("--check", action="store_true", help="Check current Python environment")
    parser.add_argument("--create-venv", type=Path, default=None, help="Create dedicated venv and install deps")
    parser.add_argument("--with-3mf", action="store_true", help="Also install optional lxml for .3MF export")
    args = parser.parse_args(argv)

    if args.create_venv is not None:
        return create_venv(args.create_venv, install_lxml=args.with_3mf)

    ok, lines = check_environment()
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
