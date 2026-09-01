"""
Lightweight stage logging for long Aristo / gmsh runs.

Enable with ``ARISTO_VERBOSE=1`` (default). Set ``ARISTO_VERBOSE=0`` to silence.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator


def aristo_verbose() -> bool:
    return os.environ.get("ARISTO_VERBOSE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def aristo_log(message: str) -> None:
    if not aristo_verbose():
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[Aristo {stamp}] {message}", flush=True)


@contextmanager
def aristo_stage(name: str) -> Iterator[None]:
    """Log START/DONE with elapsed seconds for a pipeline stage."""
    if not aristo_verbose():
        yield
        return
    aristo_log(f"START {name}")
    t0 = time.monotonic()
    try:
        yield
    except Exception as exc:
        elapsed = time.monotonic() - t0
        aristo_log(f"FAIL  {name} ({elapsed:.1f}s): {exc}")
        raise
    else:
        elapsed = time.monotonic() - t0
        aristo_log(f"DONE  {name} ({elapsed:.1f}s)")


def gmsh_terminal_enabled() -> bool:
    """Mirror mesh progress to the terminal when verbose."""
    return aristo_verbose() and os.environ.get("ARISTO_GMSH_SILENT", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )
